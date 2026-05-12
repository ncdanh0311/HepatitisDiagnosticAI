"""
backend/app/api/v1/endpoints/auth.py
======================================
Auth endpoints: register, login, refresh, me.
"""
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_user_id
from app.core.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
)
from app.core.config import settings
from app.db.session import get_db
from app.models.user import Role, User
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserProfile

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserProfile, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    # Check duplicate email
    existing = await db.scalar(select(User).where(User.email == body.email))
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Resolve role
    role = await db.scalar(select(Role).where(Role.name == body.role))
    if not role:
        raise HTTPException(status_code=400, detail=f"Role '{body.role}' not found")

    user = User(
        email=body.email,
        full_name=body.full_name,
        hashed_password=hash_password(body.password),
        role_id=role.id,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return UserProfile(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        role=role.name,
        is_active=user.is_active,
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    user = await db.scalar(select(User).where(User.email == body.email))
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")

    role = await db.scalar(select(Role).where(Role.id == user.role_id))
    role_name = role.name if role else "researcher"

    access_token = create_access_token(str(user.id), role_name)
    refresh_token = create_refresh_token(str(user.id))

    # Set refresh token as httpOnly cookie
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=not settings.DEBUG,
        samesite="lax",
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
    )

    return TokenResponse(
        access_token=access_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.get("/me", response_model=UserProfile)
async def me(
    payload: dict = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    user = await db.get(User, payload["sub"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    role = await db.scalar(select(Role).where(Role.id == user.role_id))
    return UserProfile(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        role=role.name if role else "researcher",
        is_active=user.is_active,
    )

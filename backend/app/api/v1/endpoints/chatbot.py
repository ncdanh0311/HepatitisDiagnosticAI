"""
backend/app/api/v1/endpoints/chatbot.py
========================================
Chatbot endpoint — proxies to RAG service and persists logs.
"""
import uuid

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.dependencies import get_current_user_id
from app.db.session import get_db
from app.models.chat_log import AIChatLog

router = APIRouter(prefix="/chat", tags=["chatbot"])


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[dict] = []


@router.post("/message", response_model=ChatResponse)
async def send_message(
    body: ChatRequest,
    payload: dict = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    user_id = uuid.UUID(payload["sub"])

    # Fetch session history (last 10 turns)
    history = (
        await db.scalars(
            select(AIChatLog)
            .where(AIChatLog.session_id == body.session_id)
            .order_by(AIChatLog.created_at.desc())
            .limit(10)
        )
    ).all()
    history_payload = [
        {"role": h.role, "content": h.content} for h in reversed(history)
    ]

    # Call RAG service
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            rag_resp = await client.post(
                f"{settings.RAG_SERVICE_URL}/chat",
                json={"message": body.message, "history": history_payload},
            )
            rag_resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"RAG service error: {exc}")

    rag_data = rag_resp.json()

    # Persist user turn
    db.add(AIChatLog(
        user_id=user_id, session_id=body.session_id,
        role="user", content=body.message,
    ))
    # Persist assistant turn
    db.add(AIChatLog(
        user_id=user_id, session_id=body.session_id,
        role="assistant", content=rag_data["answer"],
        sources=rag_data.get("sources", []),
    ))
    await db.commit()

    return ChatResponse(
        session_id=body.session_id,
        answer=rag_data["answer"],
        sources=rag_data.get("sources", []),
    )


@router.get("/sessions/{session_id}", response_model=list[dict])
async def get_session(
    session_id: str,
    payload: dict = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    rows = (
        await db.scalars(
            select(AIChatLog)
            .where(AIChatLog.session_id == session_id)
            .order_by(AIChatLog.created_at)
        )
    ).all()
    return [{"role": r.role, "content": r.content, "created_at": str(r.created_at)} for r in rows]

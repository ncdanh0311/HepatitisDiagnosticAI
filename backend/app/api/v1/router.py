"""
backend/app/api/v1/router.py
=============================
Central v1 API router — includes all endpoint routers.
"""
from fastapi import APIRouter

from app.api.v1.endpoints.auth import router as auth_router
from app.api.v1.endpoints.patients import router as patients_router
from app.api.v1.endpoints.predictions import router as predictions_router
from app.api.v1.endpoints.analytics import router as analytics_router
from app.api.v1.endpoints.chatbot import router as chatbot_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(auth_router)
api_router.include_router(patients_router)
api_router.include_router(predictions_router)
api_router.include_router(analytics_router)
api_router.include_router(chatbot_router)

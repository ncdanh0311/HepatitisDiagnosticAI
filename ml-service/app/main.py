"""
ml-service/app/main.py
=======================
ML Service FastAPI app (runs on port 8001).
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints.predict import router as predict_router

app = FastAPI(title="Hepatitis ML Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)

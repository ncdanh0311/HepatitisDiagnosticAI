"""
rag-service/app/api/endpoints/chat.py
=======================================
POST /chat — accept a message and return RAG-generated answer.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.rag_chain import RAGChain

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict] = []


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    try:
        chain = RAGChain.get()
        result = chain.query(body.message, body.history)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return ChatResponse(answer=result["answer"], sources=result["sources"])


@router.get("/health")
async def health():
    return {"status": "ok"}

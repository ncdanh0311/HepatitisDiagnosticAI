"""
rag-service/app/services/rag_chain.py
=======================================
LangChain RAG pipeline: embed query → retrieve from Qdrant → generate with Gemini.
"""
from __future__ import annotations

import os
from typing import Any

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Qdrant
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "hepatitis_docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

SYSTEM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a medical AI assistant specializing in hepatitis and liver diseases.
Use the following retrieved medical knowledge to answer the question accurately and concisely.
If you cannot find relevant information, say so honestly.

Context:
{context}

Question: {question}

Answer (in the same language as the question):""",
)


class RAGChain:
    _instance: "RAGChain | None" = None

    def __init__(self) -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.client = QdrantClient(url=QDRANT_URL)
        self.vectorstore = Qdrant(
            client=self.client,
            collection_name=COLLECTION,
            embeddings=self.embeddings,
        )
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.3,
        )
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": SYSTEM_PROMPT},
            return_source_documents=True,
        )

    @classmethod
    def get(cls) -> "RAGChain":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def query(self, message: str, history: list[dict]) -> dict[str, Any]:
        """
        Args:
            message: user's current question
            history: list of {"role": "user"|"assistant", "content": "..."}
        Returns:
            {"answer": str, "sources": list[dict]}
        """
        result = self.chain.invoke({"query": message})
        sources = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "snippet": doc.page_content[:200],
            }
            for doc in result.get("source_documents", [])
        ]
        return {"answer": result["result"], "sources": sources}

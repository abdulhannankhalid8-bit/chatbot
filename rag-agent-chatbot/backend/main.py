"""
RAG + Agentic AI Chatbot — FastAPI backend.

Run from project root:
    uvicorn backend.main:app --reload
Open http://localhost:8000
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.agent import agent_stream, DEFAULT_MODEL
from backend.rag import (
    DOCS_DIR,
    collection_stats,
    ingest_file,
    ingest_folder,
    search_documents,
)

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

app = FastAPI(title="RAG Agent Chatbot", version="1.0")

MODELS = {
    "openai/gpt-oss-20b": "GPT-OSS 20B · fastest",
    "openai/gpt-oss-120b": "GPT-OSS 120B · smartest",
    "qwen/qwen3.8-27b": "Qwen 3.8 27B · strong",
}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    model: str = DEFAULT_MODEL
    history: list[dict] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@app.get("/api/health")
def health():
    key = os.getenv("GROQ_API_KEY", "").strip()
    stats = collection_stats()
    return {
        "ok": True,
        "has_key": bool(key) and key != "gsk_your_key_here",
        "chunks": stats["count"],
    }


@app.get("/api/models")
def models():
    return {
        "models": [{"id": k, "label": v} for k, v in MODELS.items()],
        "default": DEFAULT_MODEL if DEFAULT_MODEL in MODELS else next(iter(MODELS)),
    }


@app.get("/api/stats")
def stats():
    return collection_stats()


@app.post("/api/ingest")
def ingest_all():
    """Re-ingest everything under data/docs."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    results = ingest_folder(DOCS_DIR)
    return results


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Upload one PDF/TXT/MD and ingest it immediately."""
    allowed = {".pdf", ".txt", ".md", ".markdown", ".csv"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported type. Use: {', '.join(allowed)}")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    dest = DOCS_DIR / Path(file.filename).name
    content = await file.read()
    dest.write_bytes(content)

    try:
        n = ingest_file(dest)
    except Exception as e:
        raise HTTPException(500, f"Ingest failed: {e}") from e

    return {"file": dest.name, "chunks": n, "stats": collection_stats()}


@app.post("/api/search")
def search(body: SearchRequest):
    """Direct RAG search (no agent) — useful for debugging."""
    hits = search_documents(body.query, top_k=body.top_k)
    return {"query": body.query, "hits": hits}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    Agentic RAG chat with SSE streaming.
    Events: status | tool | delta | done | error
    """
    if req.model not in MODELS:
        # allow anyway if user passes a valid Groq id, but warn via default list
        pass

    history = []
    for m in req.history[-12:]:  # keep short
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and content:
            history.append({"role": role, "content": content})

    async def event_stream():
        async for event in agent_stream(
            user_message=req.message.strip(),
            history=history,
            model=req.model or DEFAULT_MODEL,
        ):
            yield sse(event)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Serve frontend last
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")

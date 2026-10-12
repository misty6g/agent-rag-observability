"""FastAPI service: POST /ask -> answer + per-request telemetry. Used by Dockerfile / Cloud Run."""
from __future__ import annotations
import os
from fastapi import FastAPI
from pydantic import BaseModel
from .corpus import load_docs, build_chunks
from .retrieval import BM25Retriever, DenseRetriever, HybridRetriever, st_embedder
from .agent import RAGAgent
from .llm import HFChatLLM, EchoLLM

app = FastAPI(title="agent-rag-observability")
_agent = None

def agent() -> RAGAgent:
    global _agent
    if _agent is None:
        chunks = build_chunks(load_docs(os.environ.get("CORPUS_DIR", "data/corpus")), 200, 40)
        bm = BM25Retriever(chunks)
        if os.environ.get("RAG_LIGHT") == "1":   # no model downloads: BM25 + echo LLM (smoke test)
            _agent = RAGAgent(bm, EchoLLM())
        else:
            _agent = RAGAgent(HybridRetriever(bm, DenseRetriever(chunks, st_embedder())),
                              HFChatLLM(os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")))
    return _agent

class Ask(BaseModel):
    question: str

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/ask")
def ask(body: Ask):
    return agent().ask(body.question)

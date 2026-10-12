from __future__ import annotations
import re
import numpy as np
from rank_bm25 import BM25Okapi
from .corpus import Chunk

def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", s.lower())

class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([_tok(c.text) for c in chunks])
    def search(self, q: str, k: int) -> list[int]:
        s = self.bm25.get_scores(_tok(q))
        return [int(i) for i in np.argsort(-s, kind="stable")[:k]]

class DenseRetriever:
    """FAISS inner-product index over normalized sentence embeddings."""
    def __init__(self, chunks: list[Chunk], embedder):
        import faiss
        self.chunks, self.embedder = chunks, embedder
        X = np.asarray(embedder([c.text for c in chunks]), dtype="float32")
        self.index = faiss.IndexFlatIP(X.shape[1])
        self.index.add(X)
    def search(self, q: str, k: int) -> list[int]:
        v = np.asarray(self.embedder([q]), dtype="float32")
        _, I = self.index.search(v, k)
        return [int(i) for i in I[0] if i >= 0]

def rrf(rankings: list[list[int]], k: int, c: int = 60) -> list[int]:
    score: dict[int, float] = {}
    for r in rankings:
        for rank, idx in enumerate(r):
            score[idx] = score.get(idx, 0.0) + 1.0 / (c + rank + 1)
    return [i for i, _ in sorted(score.items(), key=lambda x: (-x[1], x[0]))][:k]

class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, pool: int = 20):
        self.chunks, self.bm25, self.dense, self.pool = dense.chunks, bm25, dense, pool
    def search(self, q: str, k: int) -> list[int]:
        return rrf([self.bm25.search(q, self.pool), self.dense.search(q, self.pool)], k)

def st_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name, device="cpu")
    return lambda texts: m.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)

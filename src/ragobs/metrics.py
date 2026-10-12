from __future__ import annotations
import re

def recall_at_k(retrieved_sources: list[str], gold: str, k: int) -> float:
    return 1.0 if gold in retrieved_sources[:k] else 0.0

def reciprocal_rank(retrieved_sources: list[str], gold: str) -> float:
    for i, s in enumerate(retrieved_sources):
        if s == gold:
            return 1.0 / (i + 1)
    return 0.0

def _norm(s: str) -> str:
    return re.sub(r"[\s`'\"*]+", " ", s.lower()).strip()

def answer_correct(answer: str, golds: list[str]) -> bool:
    a = _norm(answer)
    return any(_norm(g) in a for g in golds)

def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    pos = (len(s) - 1) * p / 100.0
    lo = int(pos); hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)

from __future__ import annotations
import json, re
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    text: str

def load_docs(corpus_dir: str | Path) -> dict[str, str]:
    d = Path(corpus_dir)
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(d.glob("*.rst"))}

def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be < size")
    words = re.findall(r"\S+", text)
    out, step = [], size - overlap
    for start in range(0, max(len(words), 1), step):
        piece = words[start:start + size]
        if piece:
            out.append(" ".join(piece))
        if start + size >= len(words):
            break
    return out

def build_chunks(docs: dict[str, str], size: int, overlap: int) -> list[Chunk]:
    chunks = []
    for name, text in docs.items():
        for i, t in enumerate(chunk_text(text, size, overlap)):
            chunks.append(Chunk(f"{name}#{i}", name, t))
    return chunks

def load_eval(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

"""Retrieval-only eval: recall@k and MRR for several configs (deterministic)."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ragobs.corpus import load_docs, build_chunks, load_eval
from ragobs.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, st_embedder
from ragobs.metrics import recall_at_k, reciprocal_rank

ROOT = Path(__file__).resolve().parents[1]
docs, ev, emb = load_docs(ROOT/"data/corpus"), load_eval(ROOT/"data/eval.jsonl"), st_embedder()
out = {}
for size, ov in [(100, 20), (200, 40), (400, 80)]:
    chunks = build_chunks(docs, size, ov)
    bm = BM25Retriever(chunks); t = time.perf_counter(); de = DenseRetriever(chunks, emb)
    for name, r in [("bm25", bm), ("dense", de), ("hybrid", HybridRetriever(bm, de))]:
        rec = {1: 0, 3: 0, 5: 0}; mrr = 0
        for q in ev:
            srcs = [chunks[i].source for i in r.search(q["question"], 10)]
            for k in rec: rec[k] += recall_at_k(srcs, q["source"], k)
            mrr += reciprocal_rank(srcs, q["source"])
        n = len(ev); key = f"{name}-c{size}"
        out[key] = {"chunks": len(chunks), **{f"recall@{k}": v/n for k, v in rec.items()}, "mrr@10": mrr/n}
        print(key, out[key], flush=True)
(ROOT/"results/retrieval.json").write_text(json.dumps(out, indent=2))

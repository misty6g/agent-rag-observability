"""End-to-end RAG eval with tracing. Resumable: appends one JSON line per question."""
import json, sys, argparse, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from ragobs.corpus import load_docs, build_chunks, load_eval
from ragobs.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, st_embedder
from ragobs.metrics import answer_correct, recall_at_k
from ragobs.agent import RAGAgent
from ragobs.llm import HFChatLLM
from ragobs.tracing import summarize, Pricing

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--configs", default="dense-c200,hybrid-c200")
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--k", type=int, default=3)
a = ap.parse_args()
random.seed(0); np.random.seed(0)
docs, ev = load_docs(ROOT/"data/corpus"), load_eval(ROOT/"data/eval.jsonl")
emb, llm = st_embedder(), HFChatLLM(a.model)
summary = {}
for cfg in a.configs.split(","):
    kind, size = cfg.split("-c"); size = int(size)
    chunks = build_chunks(docs, size, size // 5)
    bm = BM25Retriever(chunks)
    r = {"bm25": lambda: bm, "dense": lambda: DenseRetriever(chunks, emb),
         "hybrid": lambda: HybridRetriever(bm, DenseRetriever(chunks, emb))}[kind]()
    agent = RAGAgent(r, llm, k=a.k)
    log = ROOT/f"results/rag_{cfg}.jsonl"
    done = {json.loads(l)["id"] for l in log.read_text().splitlines()} if log.exists() else set()
    with log.open("a") as f:
        for q in ev:
            if q["id"] in done: continue
            res = agent.ask(q["question"])
            res.update(id=q["id"], correct=answer_correct(res["answer"], q["answers"]),
                       hit=bool(recall_at_k(res["sources"], q["source"], a.k)))
            f.write(json.dumps(res) + "\n"); f.flush()
            print(cfg, q["id"], res["correct"], round(res["latency_s"], 2), flush=True)
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    summary[cfg] = {"answer_accuracy": sum(x["correct"] for x in recs)/len(recs),
                    f"retrieval_hit@{a.k}": sum(x["hit"] for x in recs)/len(recs), **summarize(recs)}
    print(cfg, summary[cfg], flush=True)
summary["_meta"] = {"model": a.model, "k": a.k, "pricing_usd_per_1M": Pricing().__dict__,
                    "embedder": "sentence-transformers/all-MiniLM-L6-v2", "hardware": "CPU, 4 torch threads"}
(ROOT/"results/rag_summary.json").write_text(json.dumps(summary, indent=2))

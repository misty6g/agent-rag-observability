"""Rebuild results/rag_summary.json from per-question logs (results/rag_<model>_<config>.jsonl)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ragobs.tracing import summarize, Pricing
from ragobs.metrics import percentile, answer_correct
from ragobs.corpus import load_eval
R = Path(__file__).resolve().parents[1] / "results"
GOLD = {q["id"]: q["answers"] for q in load_eval(R.parent / "data/eval.jsonl")}
out = {"_pricing_assumption_usd_per_1M_tokens": Pricing().__dict__}
for f in sorted(R.glob("rag_*_*-c*.jsonl")):
    recs = [json.loads(l) for l in f.read_text().splitlines()]
    for r in recs:  # re-score from stored answers with the current (strict) matcher
        r["correct"] = answer_correct(r["answer"], GOLD[r["id"]])
    s = summarize(recs)
    s["answer_accuracy"] = sum(r["correct"] for r in recs) / len(recs)
    s["retrieval_hit@3"] = sum(r["hit"] for r in recs) / len(recs)
    s["total_tokens_per_sec_p50"] = percentile([(r["tokens_in"] + r["tokens_out"]) / r["gen_s"] for r in recs], 50)
    s["cost_per_1k_requests_usd"] = s["cost_per_request_usd_mean"] * 1000
    out[f.stem[4:]] = s
(R / "rag_summary.json").write_text(json.dumps(out, indent=2))
for k, v in out.items():
    print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})

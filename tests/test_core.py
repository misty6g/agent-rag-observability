import numpy as np, pytest
from pathlib import Path
from ragobs.corpus import chunk_text, build_chunks, load_docs, load_eval, Chunk
from ragobs.metrics import recall_at_k, reciprocal_rank, answer_correct, percentile
from ragobs.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, rrf
from ragobs.tracing import Pricing, summarize, exporter
from ragobs.agent import RAGAgent
from ragobs.llm import EchoLLM
from ragobs import pgvector_store as pg

ROOT = Path(__file__).resolve().parents[1]

def test_chunk_overlap_and_coverage():
    words = " ".join(f"w{i}" for i in range(100))
    ch = chunk_text(words, 30, 10)
    assert ch[0].split()[-10:] == ch[1].split()[:10]
    assert ch[-1].split()[-1] == "w99"

def test_chunk_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("a b", 5, 5)

def test_eval_gold_answers_present_in_gold_doc():
    docs = load_docs(ROOT / "data/corpus")
    ev = load_eval(ROOT / "data/eval.jsonl")
    assert len(ev) >= 30
    for q in ev:
        assert q["source"] in docs
        assert any(a.lower() in docs[q["source"]].lower() for a in q["answers"]), q["id"]

def test_recall_mrr():
    s = ["a", "b", "c"]
    assert recall_at_k(s, "b", 1) == 0 and recall_at_k(s, "b", 2) == 1
    assert reciprocal_rank(s, "c") == pytest.approx(1/3)
    assert reciprocal_rank(s, "z") == 0

def test_answer_correct_normalizes():
    assert answer_correct("Use `url_for()`.", ["url_for"])
    assert not answer_correct("I don't know", ["url_for"])

def test_percentile():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile(list(range(101)), 95) == 95

def test_rrf_prefers_consensus():
    assert rrf([[1, 2, 3], [2, 1, 3]], 2)[0] in (1, 2)
    assert rrf([[5, 1], [1, 6]], 1) == [1]

def _chunks():
    return [Chunk("a#0", "a", "flask session cookie name default"), Chunk("b#0", "b", "blueprint url prefix register"),
            Chunk("c#0", "c", "debug mode reloader")]

def _fake_embed(texts):
    vocab = ["session", "cookie", "blueprint", "prefix", "debug", "mode"]
    X = np.array([[t.count(w) for w in vocab] for t in texts], dtype="float32") + 1e-3
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def test_bm25_dense_hybrid_rank_correct_doc():
    c = _chunks()
    bm, de = BM25Retriever(c), DenseRetriever(c, _fake_embed)
    for r in (bm, de, HybridRetriever(bm, de)):
        assert c[r.search("blueprint prefix", 1)[0]].source == "b"

def test_pricing():
    assert Pricing(1.0, 2.0).cost(1_000_000, 500_000) == pytest.approx(2.0)

def test_agent_trace_and_telemetry():
    exporter().clear()
    ag = RAGAgent(BM25Retriever(_chunks()), EchoLLM(), k=2)
    r = ag.ask("what is the session cookie name")
    assert r["sources"][0] == "a" and r["tokens_out"] > 0 and r["cost_usd"] > 0
    names = {s.name for s in exporter().get_finished_spans()}
    assert {"rag.request", "rag.retrieve", "llm.generate"} <= names
    s = summarize([r, r])
    assert s["n"] == 2 and s["latency_p95_s"] >= s["latency_p50_s"] > 0

def test_pgvector_sql():
    assert "vector(384)" in pg.schema_sql(384)
    assert "<#>" in pg.query_sql()
    assert pg.to_pgvector([0.5, 1]) == "[0.500000,1.000000]"

def test_api_light(monkeypatch):
    monkeypatch.setenv("RAG_LIGHT", "1"); monkeypatch.setenv("CORPUS_DIR", str(ROOT / "data/corpus"))
    from fastapi.testclient import TestClient
    import ragobs.app as appmod
    appmod._agent = None
    c = TestClient(appmod.app)
    assert c.get("/healthz").json() == {"ok": True}
    r = c.post("/ask", json={"question": "default session cookie name"}).json()
    assert r["sources"] and "tokens_per_sec" in r

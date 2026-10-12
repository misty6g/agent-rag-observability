# agent-rag-observability

A retrieval-augmented QA agent over the **Flask documentation** (20 `.rst` pages from
[pallets/flask](https://github.com/pallets/flask) @ `d086db8`, BSD-3-Clause, ~33k words), with:

- **Vector store:** FAISS `IndexFlatIP` over `all-MiniLM-L6-v2` embeddings, BM25 (`rank-bm25`), and a **hybrid** (reciprocal-rank fusion of BM25 + dense).
  A pgvector / AlloyDB-compatible store (`src/ragobs/pgvector_store.py`, HNSW index, `<#>` inner-product search) is included but **was not run against a live database** — tests cover its SQL only.
- **Generator:** local `Qwen2.5-0.5B-Instruct` and `Qwen2.5-1.5B-Instruct` via Hugging Face Transformers on CPU, greedy decoding (no API keys needed).
- **Per-request tracing:** OpenTelemetry spans (`rag.request` → `rag.retrieve`, `llm.generate`) carrying latency, tokens in/out, tokens/sec and estimated cost.
- **Eval set:** 30 hand-written questions with gold answers and gold source page (`data/eval.jsonl`); a test checks every gold answer actually appears in its gold page.
- **Serving:** FastAPI (`POST /ask`, `GET /healthz`), Dockerfile, Cloud Run service spec + Cloud Build config in `deploy/`.

## Results (CPU box, 4 torch threads; all numbers produced by the scripts below)

### Retrieval (30 questions, doc-level relevance; `scripts/run_retrieval_eval.py`)

| config | chunks | recall@1 | recall@3 | recall@5 | MRR@10 |
|---|---|---|---|---|---|
| bm25-c100 | 352 | 0.767 | 0.900 | 0.900 | 0.836 |
| dense-c100 | 352 | 0.667 | 0.833 | 0.933 | 0.781 |
| hybrid-c100 | 352 | 0.767 | 0.900 | **0.967** | **0.848** |
| bm25-c200 | 178 | **0.800** | 0.833 | 0.900 | 0.848 |
| dense-c200 | 178 | 0.667 | 0.867 | 0.933 | 0.776 |
| hybrid-c200 | 178 | 0.700 | 0.867 | 0.933 | 0.798 |
| bm25-c400 | 93 | 0.700 | **0.933** | 0.933 | 0.808 |
| dense-c400 | 93 | 0.633 | 0.867 | 0.967 | 0.761 |
| hybrid-c400 | 93 | 0.733 | 0.867 | 0.967 | 0.825 |

`cN` = N-word chunks with 20% overlap. Hybrid matches or beats dense-only on MRR at every chunk size.

### End-to-end RAG with tracing (k=3, `scripts/run_rag_eval.py`, `scripts/summarize_results.py`)

| model | config | answer acc. | hit@3 | latency p50 / p95 (s) | retrieval p50 (ms) | tokens in / out (mean) | total tok/s p50 | output tok/s p50 | est. cost / 1k req |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B (fp32) | dense-c200 | 10.0% | 86.7% | 2.71 / 3.69 | 10.6 | 1026 / 6.4 | 407 | 1.59 | $0.105 |
| Qwen2.5-0.5B (fp32) | hybrid-c200 | 26.7% | 86.7% | 2.65 / 3.78 | 9.4 | 1026 / 7.7 | 411 | 1.98 | $0.106 |
| Qwen2.5-1.5B (bf16) | dense-c200 | 43.3% | 86.7% | 3.80 / 4.88 | 9.9 | 1026 / 6.2 | 289 | 1.47 | $0.105 |
| Qwen2.5-1.5B (bf16) | hybrid-c200 | **70.0%** | 86.7% | 3.41 / 5.89 | 11.9 | 1026 / 6.4 | 298 | 1.32 | $0.105 |

- **Answer accuracy** = a gold string appears as a whole token in the answer (so `SESSION_COOKIE_NAME` does not count for gold `session`).
- **Tokens/sec:** `total` = (prompt + generated tokens) / generate() wall time; `output` = generated tokens / same wall time. Answers are ~6 tokens after a ~1k-token prompt, so prefill dominates and output tok/s is low — that is a real property of short-answer RAG on CPU, not decode speed.
- **Cost pricing assumption:** $0.10 per 1M input tokens and $0.40 per 1M output tokens (a "flash/mini"-tier hosted-API price chosen as an assumption, *not* a quote and not the cost of this CPU run). Configurable in `ragobs.tracing.Pricing`.
- Hybrid retrieval raised 1.5B answer accuracy from 43.3% to 70.0% even though doc-level hit@3 is the same 86.7%: fusion surfaces the specific chunk containing the answer more often.

## Run it

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
python -m pytest -q                                  # 12 tests, no model downloads
python scripts/run_retrieval_eval.py
python scripts/run_rag_eval.py --model Qwen/Qwen2.5-1.5B-Instruct --dtype bfloat16   # resumable
python scripts/summarize_results.py
PYTHONPATH=src uvicorn ragobs.app:app --port 8080    # RAG_LIGHT=1 for BM25 + echo-LLM smoke mode
```

### Cloud Run
`deploy/cloudrun-service.yaml` (4 vCPU / 8 GiB, concurrency 1, startup probe on `/healthz`) and `deploy/cloudbuild.yaml`.
**This service has not been deployed to Cloud Run, and the Docker image has not been built** (no Docker daemon on the dev box); the FastAPI app itself was run locally with uvicorn and answered `/healthz` and `/ask`.

## Limitations
- 30 questions is small; one question = 3.3 points of accuracy. Questions were written by the author from the docs, so they favor lexical overlap (which helps BM25).
- Retrieval relevance is page-level, not passage-level.
- String-match accuracy can't credit correct paraphrases, and can't penalize a correct string inside a wrong sentence.
- The two models ran at different precisions (0.5B fp32, 1.5B bf16) because of box memory, so the latency columns are not a clean model-size comparison.
- Latency comes from a shared CPU box and includes noise from other processes.
- pgvector/AlloyDB path is untested against a live database.

Seeds are fixed (`torch.manual_seed(0)`, greedy decoding, deterministic retrieval). Per-question logs are in `results/*.jsonl`.

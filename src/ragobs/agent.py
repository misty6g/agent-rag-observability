from __future__ import annotations
import time
from .tracing import tracer, Pricing

SYSTEM = ("You answer questions about the Flask web framework using ONLY the provided context. "
          "Reply with a short, direct answer (a few words or one sentence). "
          "If the context does not contain the answer, reply 'I don't know'.")

class RAGAgent:
    def __init__(self, retriever, llm, k: int = 4, pricing: Pricing = Pricing(), max_ctx_words: int = 900):
        self.retriever, self.llm, self.k, self.pricing, self.max_ctx_words = retriever, llm, k, pricing, max_ctx_words

    def ask(self, question: str) -> dict:
        with tracer.start_as_current_span("rag.request") as span:
            t0 = time.perf_counter()
            with tracer.start_as_current_span("rag.retrieve") as rs:
                idx = self.retriever.search(question, self.k)
                rs.set_attribute("rag.k", self.k)
            t1 = time.perf_counter()
            chunks = [self.retriever.chunks[i] for i in idx]
            ctx, used = [], 0
            for c in chunks:
                w = c.text.split()
                take = w[: max(0, self.max_ctx_words - used)]
                if not take:
                    break
                ctx.append(f"[{c.source}] " + " ".join(take)); used += len(take)
            user = f"Question: {question}\n\nContext:\n" + "\n\n".join(ctx)
            with tracer.start_as_current_span("llm.generate") as gs:
                g = self.llm.generate(SYSTEM, user)
                gs.set_attribute("llm.model", getattr(self.llm, "name", "?"))
                gs.set_attribute("llm.tokens_in", g.tokens_in)
                gs.set_attribute("llm.tokens_out", g.tokens_out)
            lat = time.perf_counter() - t0
            cost = self.pricing.cost(g.tokens_in, g.tokens_out)
            tps = g.tokens_out / g.seconds if g.seconds > 0 else 0.0
            for k_, v in {"rag.latency_s": lat, "llm.tokens_per_sec": tps, "rag.cost_usd": cost}.items():
                span.set_attribute(k_, v)
            return {"answer": g.text, "sources": [c.source for c in chunks], "chunk_ids": [c.id for c in chunks],
                    "latency_s": lat, "retrieval_s": t1 - t0, "gen_s": g.seconds,
                    "tokens_in": g.tokens_in, "tokens_out": g.tokens_out, "tokens_per_sec": tps, "cost_usd": cost}

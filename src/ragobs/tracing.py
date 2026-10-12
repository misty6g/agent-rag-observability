"""Per-request OpenTelemetry tracing + cost model."""
from __future__ import annotations
from dataclasses import dataclass
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from .metrics import percentile

@dataclass(frozen=True)
class Pricing:
    """USD per 1M tokens. ASSUMPTION, not a quote: a hosted small-model list price
    in the range Google/OpenAI publish for 'flash/mini' tiers. Override via env/CLI."""
    input_per_m: float = 0.10
    output_per_m: float = 0.40
    def cost(self, tin: int, tout: int) -> float:
        return tin * self.input_per_m / 1e6 + tout * self.output_per_m / 1e6

_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
tracer = _provider.get_tracer("ragobs")

def exporter() -> InMemorySpanExporter:
    return _exporter

def summarize(records: list[dict]) -> dict:
    lat = [r["latency_s"] for r in records]
    tps = [r["tokens_out"] / r["gen_s"] for r in records if r["gen_s"] > 0]
    return {
        "n": len(records),
        "latency_p50_s": percentile(lat, 50),
        "latency_p95_s": percentile(lat, 95),
        "retrieval_p50_ms": percentile([r["retrieval_s"] * 1000 for r in records], 50),
        "tokens_in_mean": sum(r["tokens_in"] for r in records) / max(len(records), 1),
        "tokens_out_mean": sum(r["tokens_out"] for r in records) / max(len(records), 1),
        "tokens_per_sec_p50": percentile(tps, 50),
        "cost_per_request_usd_mean": sum(r["cost_usd"] for r in records) / max(len(records), 1),
    }

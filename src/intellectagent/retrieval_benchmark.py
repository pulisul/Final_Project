from __future__ import annotations

import random
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Callable

from .chunk_strategies import ChunkStrategyConfig, chunk_with_strategy
from .embeddings import HashEmbedder
from .evaluation import hit_rate_at_k, mrr
from .models import Document, EvalCase
from .reranking import safe_rerank
from .retrieval import NoopReranker
from .vectorstore import InMemoryVectorStore


@dataclass(frozen=True)
class RetrievalBenchmarkConfig:
    experiment_name: str
    seed: int = 20260831
    top_k: int = 5
    candidate_k: int = 10
    chunk_strategy: str = "semantic"
    chunk_size: int = 700
    overlap: int = 80
    embedder_name: str = "hash-smoke"
    reranker_name: str = "none"

    def validate(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be >= top_k")


def run_retrieval_benchmark(
    docs: list[Document],
    cases: list[EvalCase],
    cfg: RetrievalBenchmarkConfig,
    *,
    embedder_factory: Callable[[], object] | None = None,
    reranker_factory: Callable[[], object] | None = None,
) -> dict:
    cfg.validate()
    random.seed(cfg.seed)
    embedder = embedder_factory() if embedder_factory else HashEmbedder()
    reranker = reranker_factory() if reranker_factory else NoopReranker()
    store = InMemoryVectorStore()
    strategy = ChunkStrategyConfig(name=cfg.chunk_strategy, size=cfg.chunk_size, overlap=cfg.overlap)
    chunks = [c for d in docs for c in chunk_with_strategy(d, strategy)]
    store.upsert(chunks, embedder.embed([c.text for c in chunks]) if chunks else [])

    pairs = []
    rows = []
    search_latencies: list[float] = []
    rerank_latencies: list[float] = []
    total_latencies: list[float] = []
    reranker_fallbacks = 0

    for case in cases:
        t_total = time.perf_counter()
        t0 = time.perf_counter()
        qv = embedder.embed([case.question])[0]
        candidates = store.query(qv, cfg.candidate_k)
        search_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        rr = safe_rerank(case.question, candidates, reranker)
        rerank_ms = (time.perf_counter() - t1) * 1000
        if rr.fallback_used:
            reranker_fallbacks += 1
        hits = rr.hits[: cfg.top_k]
        total_ms = (time.perf_counter() - t_total) * 1000

        search_latencies.append(search_ms)
        rerank_latencies.append(rerank_ms)
        total_latencies.append(total_ms)
        pairs.append((case, hits))
        rows.append({
            "case_id": case.case_id,
            "question": case.question,
            "relevant_doc_ids": list(case.relevant_doc_ids),
            "candidate_doc_ids": [h.doc_id for h in candidates],
            "hit_doc_ids": [h.doc_id for h in hits],
            "hit_chunk_ids": [h.chunk_id for h in hits],
            "search_ms": search_ms,
            "rerank_ms": rerank_ms,
            "total_ms": total_ms,
            "reranker_fallback": rr.fallback_used,
            "reranker_error": rr.error,
        })

    def mean(xs: list[float]) -> float:
        return statistics.mean(xs) if xs else 0.0

    def p95(xs: list[float]) -> float:
        return sorted(xs)[max(0, int(len(xs) * 0.95) - 1)] if xs else 0.0

    return {
        "config": asdict(cfg),
        "document_count": len(docs),
        "chunk_count": len(chunks),
        "case_count": len(cases),
        "metrics": {
            "hit_rate_at_k": hit_rate_at_k(pairs),
            "mrr": mrr(pairs),
            "search_ms_mean": mean(search_latencies),
            "rerank_ms_mean": mean(rerank_latencies),
            "latency_ms_mean": mean(total_latencies),
            "latency_ms_p95": p95(total_latencies),
            "reranker_fallback_rate": reranker_fallbacks / len(cases) if cases else 0.0,
        },
        "cases": rows,
    }

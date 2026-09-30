from __future__ import annotations
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .chunk_strategies import ChunkStrategyConfig, chunk_with_strategy
from .embeddings import HashEmbedder
from .evaluation import hit_rate_at_k, mrr
from .models import Document, EvalCase
from .retrieval import Retriever
from .vectorstore import InMemoryVectorStore


@dataclass(frozen=True)
class RetrievalExperimentConfig:
    experiment_name: str
    seed: int
    top_k: int
    strategy: ChunkStrategyConfig
    embedder_name: str = "hash-smoke"


def save_experiment_config(path: Path, cfg: RetrievalExperimentConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")


def run_retrieval_experiment(
    docs: list[Document],
    cases: list[EvalCase],
    cfg: RetrievalExperimentConfig,
    embedder_factory: Callable[[], object] | None = None,
) -> dict:
    random.seed(cfg.seed)
    embedder = embedder_factory() if embedder_factory else HashEmbedder()
    store = InMemoryVectorStore()
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_with_strategy(doc, cfg.strategy))
    vectors = embedder.embed([c.text for c in all_chunks]) if all_chunks else []
    store.upsert(all_chunks, vectors)
    retriever = Retriever(embedder, store)

    pairs = []
    latencies = []
    rows = []
    for case in cases:
        t0 = time.perf_counter()
        hits = retriever.search(case.question, top_k=cfg.top_k)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)
        pairs.append((case, hits))
        rows.append({
            "case_id": case.case_id,
            "question": case.question,
            "relevant_doc_ids": list(case.relevant_doc_ids),
            "hit_doc_ids": [h.doc_id for h in hits],
            "hit_chunk_ids": [h.chunk_id for h in hits],
            "latency_ms": elapsed,
        })

    avg = sum(latencies) / len(latencies) if latencies else 0.0
    p95 = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)] if latencies else 0.0
    return {
        "config": asdict(cfg),
        "document_count": len(docs),
        "chunk_count": len(all_chunks),
        "case_count": len(cases),
        "metrics": {
            "hit_rate_at_k": hit_rate_at_k(pairs),
            "mrr": mrr(pairs),
            "latency_ms_mean": avg,
            "latency_ms_p95": p95,
        },
        "cases": rows,
    }

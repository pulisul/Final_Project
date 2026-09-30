from pathlib import Path

import pytest

from intellectagent.models import Document, EvalCase
from intellectagent.retrieval_benchmark import RetrievalBenchmarkConfig, run_retrieval_benchmark
from intellectagent.reranking import FailingReranker, TokenOverlapReranker


def docs():
    return [
        Document("d1", Path("a.md"), "사과는 과일입니다.", "a"),
        Document("d2", Path("b.md"), "자동차는 이동수단입니다.", "b"),
    ]


def cases():
    return [EvalCase("q1", "사과는 무엇인가요?", ("d1",), False)]


def test_candidate_k_must_cover_top_k():
    cfg = RetrievalBenchmarkConfig("bad", top_k=5, candidate_k=3)
    with pytest.raises(ValueError):
        cfg.validate()


def test_benchmark_returns_quality_and_latency():
    cfg = RetrievalBenchmarkConfig("ok", top_k=1, candidate_k=2)
    r = run_retrieval_benchmark(docs(), cases(), cfg, reranker_factory=TokenOverlapReranker)
    assert 0 <= r["metrics"]["hit_rate_at_k"] <= 1
    assert r["metrics"]["latency_ms_mean"] >= 0


def test_benchmark_survives_reranker_failure():
    cfg = RetrievalBenchmarkConfig("fail", top_k=1, candidate_k=2)
    r = run_retrieval_benchmark(docs(), cases(), cfg, reranker_factory=FailingReranker)
    assert r["metrics"]["reranker_fallback_rate"] == 1.0

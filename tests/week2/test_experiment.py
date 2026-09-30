from pathlib import Path
from intellectagent.chunk_strategies import ChunkStrategyConfig
from intellectagent.experiment import RetrievalExperimentConfig, run_retrieval_experiment
from intellectagent.hashing import sha256_text
from intellectagent.models import Document, EvalCase


def test_retrieval_experiment_returns_metrics():
    d1 = Document("d1", Path("a.md"), "사과는 과일입니다.", sha256_text("a"), {"name": "a.md"})
    d2 = Document("d2", Path("b.md"), "RAG는 검색 후 생성합니다.", sha256_text("b"), {"name": "b.md"})
    cases = [
        EvalCase("q1", "사과는 무엇인가요?", ("d1",), False),
        EvalCase("q2", "RAG는 무엇을 하나요?", ("d2",), False),
    ]
    cfg = RetrievalExperimentConfig("t", 123, 2, ChunkStrategyConfig("length", size=100, overlap=10))
    r = run_retrieval_experiment([d1, d2], cases, cfg)
    assert r["case_count"] == 2
    assert 0.0 <= r["metrics"]["hit_rate_at_k"] <= 1.0
    assert 0.0 <= r["metrics"]["mrr"] <= 1.0

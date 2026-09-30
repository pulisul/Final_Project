from intellectagent.models import SearchHit
from intellectagent.reranking import FailingReranker, TokenOverlapReranker, safe_rerank


def hit(cid, text, score=0.1):
    return SearchHit(cid, "doc", text, score, {})


def test_token_overlap_reranker_prefers_matching_text():
    rr = TokenOverlapReranker().rerank("사과 과일", [hit("a", "자동차 엔진"), hit("b", "사과는 과일")])
    assert rr[0].chunk_id == "b"


def test_safe_rerank_falls_back_on_failure():
    hits = [hit("a", "x", 0.9), hit("b", "y", 0.8)]
    rr = safe_rerank("q", hits, FailingReranker())
    assert rr.fallback_used is True
    assert [x.chunk_id for x in rr.hits] == ["a", "b"]
    assert "RuntimeError" in (rr.error or "")

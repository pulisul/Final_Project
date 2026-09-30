from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .models import SearchHit


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w가-힣]+", text.lower()))


@dataclass(frozen=True)
class RerankResult:
    hits: list[SearchHit]
    fallback_used: bool
    error: str | None = None


class TokenOverlapReranker:
    """Dependency-free deterministic reranker used only for tests/smoke experiments."""

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        q = _tokens(query)
        ranked: list[SearchHit] = []
        for hit in hits:
            h = _tokens(hit.text)
            overlap = len(q.intersection(h))
            denom = max(1, len(q))
            score = overlap / denom
            ranked.append(
                SearchHit(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    text=hit.text,
                    score=float(score),
                    metadata={**hit.metadata, "reranker": "token-overlap"},
                )
            )
        return sorted(ranked, key=lambda x: (x.score, x.chunk_id), reverse=True)


class FailingReranker:
    """Test double used to prove retrieval falls back when reranking fails."""

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        raise RuntimeError("intentional reranker failure")


def safe_rerank(query: str, hits: list[SearchHit], reranker) -> RerankResult:
    try:
        return RerankResult(list(reranker.rerank(query, hits)), False, None)
    except Exception as exc:  # enhancement must never break core retrieval
        return RerankResult(list(hits), True, f"{type(exc).__name__}: {exc}")

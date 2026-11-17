"""'근거 없음' 판정에 쓰는 점수(gate)를 한 곳에서 정한다.

CoreRAG 와 AgenticRAGWorkflow(LangGraph 노드)가 같은 값을 써야 한다. 예전에는 워크플로가 ``hits[0].score`` 만 봐서,
융합 점수(RRF ≈ 0.03)를 쓰는 하이브리드 검색에서는 벡터 코사인이 충분해도 항상 거절로 갔다.
"""
from __future__ import annotations

from typing import Sequence

from .models import SearchHit


def gate_score(hits: Sequence[SearchHit]) -> float:
    """하이브리드/BGE-M3 결과는 ``metadata['gate_score']``(벡터 코사인 원점수)의 최댓값, 없으면 1순위 점수. 결과가 없으면 0.0."""
    if not hits:
        return 0.0
    gs = [h.metadata["gate_score"] for h in hits if h.metadata.get("gate_score") is not None]
    if any("gate_score" in h.metadata for h in hits):
        return max(gs) if gs else hits[0].score
    return hits[0].score

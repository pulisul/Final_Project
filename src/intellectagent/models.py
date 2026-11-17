from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class Document:
    doc_id: str
    path: Path
    text: str
    sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    strategy: str
    ordinal: int
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    citations: tuple[str, ...]
    hits: tuple[SearchHit, ...]
    fallback: bool
    policy: str = "baseline"

@dataclass(frozen=True)
class EvalCase:
    case_id: str
    question: str
    relevant_doc_ids: tuple[str, ...]
    should_fallback: bool = False


def index_text(chunk: "Chunk") -> str:
    """색인(임베딩·BM25)에 넣는 글. 문맥 머리말(``metadata["context"]``)이 있으면 본문 앞에 붙인다. 화면·근거·평가에는 항상 원래 ``chunk.text`` 를 쓴다."""
    ctx = chunk.metadata.get("context") if chunk.metadata else None
    return f"{ctx}\n{chunk.text}" if ctx else chunk.text

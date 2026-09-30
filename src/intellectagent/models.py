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

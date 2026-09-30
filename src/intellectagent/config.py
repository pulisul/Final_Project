from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class CoreConfig:
    project_root: Path
    allowed_root: Path
    index_dir: Path
    metadata_db: Path
    evidence_dir: Path
    collection_name: str = "intellectagent"
    max_file_bytes: int = 10 * 1024 * 1024
    allowed_extensions: tuple[str, ...] = (".md", ".txt", ".pdf")
    semantic_min_chars: int = 120
    semantic_max_chars: int = 1200
    fallback_chunk_chars: int = 700
    fallback_overlap_chars: int = 80
    top_k: int = 5
    candidate_k: int = 10
    fallback_min_score: float = 0.12
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    ollama_model: str = "qwen2.5:14b"
    runtime_profile: str = "auto"
    feature_flags: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "CoreConfig":
        path = path.resolve()
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        root = path.parent.parent.resolve()
        def p(key: str) -> Path:
            return (root / data[key]).resolve()
        return cls(
            project_root=root,
            allowed_root=p("allowed_root"),
            index_dir=p("index_dir"),
            metadata_db=p("metadata_db"),
            evidence_dir=p("evidence_dir"),
            collection_name=data.get("collection_name", "intellectagent"),
            max_file_bytes=int(data.get("max_file_bytes", 10 * 1024 * 1024)),
            allowed_extensions=tuple(data.get("allowed_extensions", [".md", ".txt", ".pdf"])),
            semantic_min_chars=int(data.get("semantic_min_chars", 120)),
            semantic_max_chars=int(data.get("semantic_max_chars", 1200)),
            fallback_chunk_chars=int(data.get("fallback_chunk_chars", 700)),
            fallback_overlap_chars=int(data.get("fallback_overlap_chars", 80)),
            top_k=int(data.get("top_k", 5)),
            candidate_k=int(data.get("candidate_k", max(5, int(data.get("top_k", 5))))),
            fallback_min_score=float(data.get("fallback_min_score", 0.12)),
            embedding_model=data.get("embedding_model", "BAAI/bge-m3"),
            reranker_model=data.get("reranker_model", "BAAI/bge-reranker-v2-m3"),
            ollama_model=data.get("ollama_model", "qwen2.5:14b"),
            runtime_profile=str(data.get("runtime_profile", "auto")),
            feature_flags=dict(data.get("feature_flags", {})),
        )

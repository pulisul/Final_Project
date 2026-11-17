from __future__ import annotations
from dataclasses import dataclass
from .chunking import _length_chunks, chunk_document
from .hashing import sha256_text
from .models import Chunk, Document

@dataclass(frozen=True)
class ChunkStrategyConfig:
    name: str
    size: int = 700
    overlap: int = 80
    min_chars: int = 120
    max_chars: int = 1200
    context_header: bool = False   # True 면 색인용 문맥 머리말을 단다(chunk.text 는 그대로)


def chunk_with_strategy(doc: Document, cfg: ChunkStrategyConfig) -> list[Chunk]:
    chunks = _chunk_with_strategy(doc, cfg)
    if cfg.context_header:
        from .context_headers import add_context_headers

        chunks = add_context_headers(doc, chunks)
    return chunks


def _chunk_with_strategy(doc: Document, cfg: ChunkStrategyConfig) -> list[Chunk]:
    """Create deterministic chunks using one named strategy.

    Supported strategies:
      - semantic: current header/paragraph-aware P0 strategy
      - length: fixed-size character windows with overlap
    """
    if cfg.name == "semantic":
        return chunk_document(
            doc,
            min_chars=cfg.min_chars,
            max_chars=cfg.max_chars,
            fallback_size=cfg.size,
            fallback_overlap=cfg.overlap,
        )
    if cfg.name == "length":
        texts = _length_chunks(doc.text, cfg.size, cfg.overlap)
        chunks: list[Chunk] = []
        for i, text in enumerate(texts):
            cid = sha256_text(f"{doc.doc_id}:length:{cfg.size}:{cfg.overlap}:{i}:{text}")[:20]
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    doc_id=doc.doc_id,
                    text=text,
                    strategy="length",
                    ordinal=i,
                    metadata={"source": doc.metadata.get("name", "")},
                )
            )
        return chunks
    raise ValueError(f"unsupported chunk strategy: {cfg.name}")

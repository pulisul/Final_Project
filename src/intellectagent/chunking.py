from __future__ import annotations
import re
from .hashing import sha256_text
from .models import Chunk, Document

def _length_chunks(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text: return []
    if overlap >= size: raise ValueError("overlap must be smaller than size")
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        out.append(text[start:end].strip())
        if end == len(text): break
        start = end - overlap
    return [x for x in out if x]

def chunk_document(doc: Document, min_chars: int = 120, max_chars: int = 1200,
                   fallback_size: int = 700, fallback_overlap: int = 80) -> list[Chunk]:
    # Header/paragraph semantic-ish segmentation without binding the core to LangChain.
    blocks = [b.strip() for b in re.split(r"(?m)(?=^#{1,6}\s)|\n\s*\n+", doc.text) if b.strip()]
    merged: list[str] = []
    buf = ""
    for block in blocks:
        candidate = (buf + "\n\n" + block).strip() if buf else block
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf: merged.append(buf)
            if len(block) > max_chars:
                merged.extend(_length_chunks(block, fallback_size, fallback_overlap))
                buf = ""
            else:
                buf = block
    if buf: merged.append(buf)
    # Merge tiny chunks forward where practical.
    normalized: list[str] = []
    for piece in merged:
        if normalized and len(piece) < min_chars and len(normalized[-1]) + len(piece) + 2 <= max_chars:
            normalized[-1] += "\n\n" + piece
        else:
            normalized.append(piece)
    if not normalized:
        normalized = _length_chunks(doc.text, fallback_size, fallback_overlap)
    chunks: list[Chunk] = []
    for i, text in enumerate(normalized):
        strategy = "semantic" if len(blocks) > 1 and len(text) <= max_chars else "length"
        cid = sha256_text(f"{doc.doc_id}:{i}:{text}")[:20]
        chunks.append(Chunk(cid, doc.doc_id, text, strategy, i, {"source": doc.metadata.get("name", "")}))
    return chunks

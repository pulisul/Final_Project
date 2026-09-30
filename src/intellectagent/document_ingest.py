from __future__ import annotations

import hashlib
import os
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .indexing import Indexer
from .loaders import SafeDocumentLoader
from .metadata import MetadataDB


class UploadValidationError(ValueError):
    """Raised when an uploaded document violates the ingestion contract."""


@dataclass(frozen=True)
class SavedUpload:
    original_name: str
    stored_name: str
    path: str
    bytes: int
    sha256: str
    duplicate: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_upload_name(name: str) -> str:
    """Return a safe basename and reject path-bearing or empty upload names."""
    if not isinstance(name, str) or not name.strip() or "\x00" in name:
        raise UploadValidationError("invalid upload filename")
    normalized = unicodedata.normalize("NFC", name.strip())
    basename = Path(normalized).name
    if basename != normalized or basename in {".", "..", ""}:
        raise UploadValidationError("path components are not allowed in upload filename")
    return basename


def save_upload_bytes(
    *,
    allowed_root: Path,
    filename: str,
    data: bytes,
    allowed_extensions: Iterable[str],
    max_file_bytes: int,
) -> SavedUpload:
    """Safely and atomically save one uploaded document below allowed_root.

    Existing identical content is treated as an idempotent duplicate. If a file with
    the same visible name has different content, a short content hash is appended so
    the old source document is never silently overwritten.
    """
    root = Path(allowed_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    safe_name = normalize_upload_name(filename)
    allowed = {str(x).lower() for x in allowed_extensions}
    suffix = Path(safe_name).suffix.lower()
    if suffix not in allowed:
        raise UploadValidationError(f"extension not allowed: {suffix or '(none)'}")
    if not data:
        raise UploadValidationError("empty files are not allowed")
    if len(data) > int(max_file_bytes):
        raise UploadValidationError(f"file too large: {len(data)} > {int(max_file_bytes)}")

    digest = _sha256_bytes(data)
    target = root / safe_name
    if target.exists():
        if not target.is_file() or target.is_symlink():
            raise UploadValidationError("upload target is not a regular file")
        current = hashlib.sha256(target.read_bytes()).hexdigest()
        if current == digest:
            return SavedUpload(filename, target.name, str(target), len(data), digest, True)
        target = root / f"{target.stem}__{digest[:8]}{target.suffix.lower()}"
        if target.exists():
            current = hashlib.sha256(target.read_bytes()).hexdigest()
            if current == digest:
                return SavedUpload(filename, target.name, str(target), len(data), digest, True)
            raise UploadValidationError("hash-suffixed upload target already exists with different content")

    resolved_parent = target.parent.resolve()
    if resolved_parent != root:
        raise UploadValidationError("upload path escaped allowed root")

    tmp = root / f".upload-{uuid.uuid4().hex}.tmp"
    try:
        with tmp.open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)

    # Reuse the production loader as a second validation layer. Invalid PDFs (e.g.
    # no extractable text) are removed instead of leaving unusable files in real_docs.
    loader = SafeDocumentLoader(root, tuple(allowed), int(max_file_bytes))
    try:
        loader.load(target)
    except Exception:
        target.unlink(missing_ok=True)
        raise

    return SavedUpload(filename, target.name, str(target), len(data), digest, False)


def list_ingestible_documents(allowed_root: Path, allowed_extensions: Iterable[str]) -> list[Path]:
    root = Path(allowed_root).resolve()
    if not root.exists():
        return []
    allowed = {str(x).lower() for x in allowed_extensions}
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and not p.is_symlink() and p.suffix.lower() in allowed
    )


def index_runtime_documents(*, cfg, embedder, store) -> dict:
    """Index cfg.allowed_root into the already-loaded runtime vector store."""
    loader = SafeDocumentLoader(cfg.allowed_root, cfg.allowed_extensions, cfg.max_file_bytes)
    db = MetadataDB(cfg.metadata_db)
    checkpoint = cfg.evidence_dir / "index_checkpoint.json"
    state = Indexer(loader, embedder, store, db, cfg).index_all(checkpoint)
    state["checkpoint_path"] = str(checkpoint)
    state["document_count"] = len(loader.discover())
    return state

from __future__ import annotations
from pathlib import Path
from .hashing import sha256_file
from .models import Document

class UnsafePathError(ValueError): pass
class UnsupportedFileError(ValueError): pass

class SafeDocumentLoader:
    def __init__(self, allowed_root: Path, allowed_extensions: tuple[str, ...], max_file_bytes: int):
        self.allowed_root = allowed_root.resolve()
        self.allowed_extensions = {e.lower() for e in allowed_extensions}
        self.max_file_bytes = max_file_bytes

    def _validate(self, path: Path) -> Path:
        candidate = path.expanduser()
        if candidate.is_symlink():
            raise UnsafePathError(f"symlink rejected: {candidate}")
        resolved = candidate.resolve(strict=True)
        if self.allowed_root != resolved and self.allowed_root not in resolved.parents:
            raise UnsafePathError(f"outside allowed root: {resolved}")
        if resolved.suffix.lower() not in self.allowed_extensions:
            raise UnsupportedFileError(f"extension not allowed: {resolved.suffix}")
        if not resolved.is_file():
            raise UnsupportedFileError(f"not a file: {resolved}")
        if resolved.stat().st_size > self.max_file_bytes:
            raise UnsupportedFileError(f"file too large: {resolved.stat().st_size}")
        return resolved

    def load(self, path: Path) -> Document:
        resolved = self._validate(path)
        suffix = resolved.suffix.lower()
        if suffix in {".md", ".txt"}:
            text = resolved.read_text(encoding="utf-8", errors="replace")
        elif suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as e:
                raise RuntimeError("pypdf is required for PDF files") from e
            reader = PdfReader(str(resolved))
            text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
            if not text.strip():
                raise UnsupportedFileError("PDF has no extractable text; OCR is out of P0 scope")
        else:
            raise UnsupportedFileError(suffix)
        digest = sha256_file(resolved)
        return Document(doc_id=digest[:16], path=resolved, text=text, sha256=digest,
                        metadata={"name": resolved.name, "suffix": suffix, "bytes": resolved.stat().st_size})

    def discover(self) -> list[Path]:
        paths: list[Path] = []
        for p in self.allowed_root.rglob("*"):
            if p.is_file() and not p.is_symlink() and p.suffix.lower() in self.allowed_extensions:
                paths.append(p)
        return sorted(paths)

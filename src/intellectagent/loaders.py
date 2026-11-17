from __future__ import annotations
import unicodedata
from pathlib import Path
from . import doc_parsers
from .hashing import sha256_bytes, sha256_file
from .models import Document
from .textnorm import clean_text

def _nfc(text: str) -> str:
    """색인 본문 정리: 자모 분리(NFD) → 완성형(NFC), 눈에 안 보이는 글자 제거. 질문 쪽과 같은 함수(textnorm)를 쓴다."""
    return clean_text(text)


def _require_text_like(text: str, suffix: str) -> None:
    """이진 파일을 .txt/.md 로 위장했을 때 쓰레기를 지식으로 색인하지 않는다.
    NUL 이 있거나 출력 가능한 글자가 90% 미만이면 거부한다(정상 한글·영문 문서는 거의 100%)."""
    sample = text[:20000]
    if not sample.strip():
        return
    if "\x00" in sample:
        raise UnsupportedFileError(f"{suffix}: 텍스트가 아닌 이진 데이터로 보임(NUL 포함)")
    # 제어문자(Cc)·미할당·사용자정의·대리쌍·대체문자(U+FFFD)만 '읽을 수 없는 글자'로 센다. 제로폭 같은 서식 문자(Cf)는 정상이다.
    bad = sum(1 for ch in sample if ch not in "\n\r\t" and (ch == "\ufffd" or unicodedata.category(ch) in ("Cc", "Cs", "Co", "Cn")))
    printable = 1 - bad / len(sample)
    if printable < 0.9:
        raise UnsupportedFileError(f"{suffix}: 텍스트가 아닌 이진 데이터로 보임(읽을 수 있는 글자 {printable:.0%})")


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
            # BOM 제거 + UTF-8 실패 시 CP949(한국어 윈도우 메모장) 로 해석. 기존: errors="replace" 로 한글이 깨졌다.
            text = doc_parsers._decode_text(resolved.read_bytes())
            _require_text_like(text, suffix)
        elif suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as e:
                raise RuntimeError("pypdf is required for PDF files") from e
            reader = PdfReader(str(resolved))
            text = "\n\n".join(doc_parsers.clean_pdf_page(page.extract_text() or "") for page in reader.pages)
            if not text.strip():
                raise UnsupportedFileError("PDF has no extractable text; OCR is out of P0 scope")
        elif suffix in doc_parsers.OFFICE_EXTENSIONS:
            # 읽은 바이트와 해시 대상을 일치시킨다(파일을 두 번 읽는 사이에 바뀌는 것을 막는다).
            data = resolved.read_bytes()
            try:
                text = doc_parsers.extract_text(resolved, data)
            except doc_parsers.ParseError as exc:
                raise UnsupportedFileError(f"{suffix}: {exc}") from exc
            if not text.strip():
                raise UnsupportedFileError(f"{suffix} has no extractable text")
            digest = sha256_bytes(data)
            return Document(doc_id=digest[:16], path=resolved, text=_nfc(text), sha256=digest,
                            metadata={"name": resolved.name, "suffix": suffix, "bytes": len(data)})
        else:
            raise UnsupportedFileError(suffix)
        digest = sha256_file(resolved)
        return Document(doc_id=digest[:16], path=resolved, text=_nfc(text), sha256=digest,
                        metadata={"name": resolved.name, "suffix": suffix, "bytes": resolved.stat().st_size})

    def discover(self) -> list[Path]:
        paths: list[Path] = []
        for p in self.allowed_root.rglob("*"):
            if p.is_file() and not p.is_symlink() and p.suffix.lower() in self.allowed_extensions and not p.name.lower().endswith(".expect.txt"):
                paths.append(p)  # '<문서>.expect.txt' 는 사람이 적은 정답 메모이므로 말뭉치에 넣지 않는다
        return sorted(paths)

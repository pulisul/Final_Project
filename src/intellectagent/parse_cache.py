"""파싱 결과 캐시 (AutoRAG 2.0 의 "parsed mirror" 아이디어).

원본 파일 -> (파싱) -> ``<cache>/<파서지문>/<해시2자>/<sha256>.<확장자>.md`` 에 *사람이 열어볼 수 있는 텍스트* 로 저장.
- 청킹/임베딩 설정만 바꿔 재실험할 때 PDF/HWP 파싱을 다시 하지 않는다.
- 파싱이 깨졌는지(표, 스캔본, 한글 인코딩) 사람이 .md 를 열어 눈으로 확인할 수 있다.

안전 설계
- 키 = 원본 *바이트* 의 SHA-256 + 확장자 + 파서 지문. 파일 내용이 바뀌거나 파서 코드가 바뀌면 자동으로 다른 키가 된다.
- 읽을 때마다 저장된 텍스트의 해시를 사이드카 JSON 과 대조한다. 누가 .md 를 손으로 고쳤거나 파일이 깨졌으면 캐시를 버리고 다시 파싱한다.
  (사람이 .md 를 고쳐서 "교정된 텍스트"를 쓰는 용도는 이 캐시의 목적이 아니다 — 변조 감지가 우선이다.)
- 경로 검증/크기 제한/확장자 허용은 SafeDocumentLoader 가 그대로 먼저 수행한다. 캐시는 그 *뒤* 에 붙는다.
- 캐시 폴더가 허용 루트 안에 있으면 미러 .md 가 원본 문서로 다시 색인되는 순환이 생긴다. ``discover()`` 가 캐시 폴더를 제외한다.
- 쓰기는 임시파일 + os.replace(원자적). 본문(.md)을 먼저, 사이드카(.json)를 나중에 쓰므로 중간에 죽어도 "없음"으로 취급된다.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from . import doc_parsers, loaders
from .hashing import sha256_bytes, sha256_text
from .models import Document

_ENC = dict(errors="surrogatepass")  # pypdf 가 드물게 내놓는 고아 서로게이트까지 무손실로 왕복


def parser_fingerprint() -> str:
    """파서 소스(doc_parsers.py, loaders.py) 와 pypdf 버전의 해시. 파서가 바뀌면 캐시가 자동 무효화된다."""
    h = [Path(doc_parsers.__file__).read_bytes(), Path(loaders.__file__).read_bytes()]
    try:
        from importlib.metadata import version

        h.append(version("pypdf").encode())
    except Exception:
        h.append(b"pypdf:none")
    return sha256_bytes(b"\0".join(h))[:12]


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    corrupt: int = 0
    writes: int = 0


class ParseCache:
    def __init__(self, cache_dir: Path, fingerprint: str | None = None):
        self.root = Path(cache_dir).expanduser().resolve()
        self.fingerprint = fingerprint or parser_fingerprint()
        self.stats = CacheStats()

    # ---- 경로 ---------------------------------------------------------------------------------------------------
    def _base(self, sha: str, suffix: str) -> Path:
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("sha must be 64 hex chars")
        ext = suffix.lower().lstrip(".")
        if not ext.isalnum():
            raise ValueError(f"bad suffix: {suffix!r}")
        return self.root / self.fingerprint / sha[:2] / f"{sha}.{ext}"

    def mirror_path(self, sha: str, suffix: str) -> Path:
        """사람이 열어볼 .md 파일 경로."""
        return self._base(sha, suffix).with_name(self._base(sha, suffix).name + ".md")

    def _meta_path(self, sha: str, suffix: str) -> Path:
        return self._base(sha, suffix).with_name(self._base(sha, suffix).name + ".json")

    # ---- 읽기/쓰기 -----------------------------------------------------------------------------------------------
    def get(self, sha: str, suffix: str) -> str | None:
        md, meta = self.mirror_path(sha, suffix), self._meta_path(sha, suffix)
        if not md.is_file() or not meta.is_file() or md.is_symlink() or meta.is_symlink():
            self.stats.misses += 1
            return None
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
            text = md.read_bytes().decode("utf-8", **_ENC)
            ok = info.get("sha256") == sha and info.get("text_sha256") == sha256_text_lossless(text)
        except Exception:
            ok = False
        if not ok:
            self.stats.corrupt += 1
            return None
        self.stats.hits += 1
        return text

    def put(self, sha: str, suffix: str, text: str, name: str = "") -> Path:
        md, meta = self.mirror_path(sha, suffix), self._meta_path(sha, suffix)
        md.parent.mkdir(parents=True, exist_ok=True)
        info = {"sha256": sha, "suffix": suffix.lower(), "name": name, "chars": len(text),
                "text_sha256": sha256_text_lossless(text), "fingerprint": self.fingerprint}
        _atomic_write(md, text.encode("utf-8", **_ENC))
        _atomic_write(meta, json.dumps(info, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"))
        self.stats.writes += 1
        return md

    # ---- 점검/정리 -----------------------------------------------------------------------------------------------
    def verify_all(self) -> list[str]:
        """현재 지문 폴더의 모든 항목을 다시 검사해 문제가 있는 .md 경로 목록을 돌려준다."""
        bad: list[str] = []
        base = self.root / self.fingerprint
        if not base.is_dir():
            return bad
        for meta in sorted(base.rglob("*.json")):
            md = meta.with_name(meta.name[:-5] + ".md")
            try:
                info = json.loads(meta.read_text(encoding="utf-8"))
                text = md.read_bytes().decode("utf-8", **_ENC)
                if info.get("text_sha256") != sha256_text_lossless(text):
                    bad.append(str(md))
            except Exception:
                bad.append(str(md))
        return bad

    def purge_other_versions(self) -> int:
        """다른 파서 지문의 폴더(= 낡은 캐시)를 지운다. 지운 폴더 수를 반환."""
        import shutil

        n = 0
        if self.root.is_dir():
            for d in self.root.iterdir():
                if d.is_dir() and not d.is_symlink() and d.name != self.fingerprint and len(d.name) == 12:
                    shutil.rmtree(d)
                    n += 1
        return n


def sha256_text_lossless(text: str) -> str:
    return sha256_bytes(text.encode("utf-8", **_ENC))


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class CachingDocumentLoader:
    """SafeDocumentLoader 를 감싸 파싱 결과를 캐시한다. ``load``/``discover`` 의 결과는 감싼 로더와 동일하다."""

    def __init__(self, loader: loaders.SafeDocumentLoader, cache: ParseCache):
        self._loader = loader
        self.cache = cache

    def __getattr__(self, name):  # allowed_root 등 나머지 속성은 그대로 위임
        return getattr(self._loader, name)

    def discover(self):
        cache_root = self.cache.root
        return [p for p in self._loader.discover() if cache_root != p.resolve() and cache_root not in p.resolve().parents]

    def load(self, path: Path) -> Document:
        resolved = self._loader._validate(path)  # 심볼릭 링크/루트 이탈/확장자/크기 검사는 캐시와 무관하게 항상 수행
        data = resolved.read_bytes()
        sha = sha256_bytes(data)
        suffix = resolved.suffix.lower()
        text = self.cache.get(sha, suffix)
        if text is not None:
            return Document(doc_id=sha[:16], path=resolved, text=text, sha256=sha,
                            metadata={"name": resolved.name, "suffix": suffix, "bytes": len(data)})
        doc = self._loader.load(path)  # 실패(ParseError 등)는 그대로 전파하고 캐시하지 않는다
        self.cache.put(doc.sha256, suffix, doc.text, name=resolved.name)
        return doc

"""여러 폴더를 안전하게 훑어 '큰' 말뭉치를 만든다 (AutoRAG ``autorag-setup`` 의 문서 루트 탐색 + dupey 중복 제외 + 갱신 감지에 해당).

AutoRAG 도 PC 를 *무작정* 전부 읽지 않는다: 승인된 루트만, 시스템/캐시/자격증명 폴더는 제외, 원본은 절대 수정·삭제하지 않는다.
여기서도 같은 원칙을 코드로 강제한다.

- 루트는 반드시 명시한다. 파일시스템 루트(``/``, ``C:\\``), 시스템 폴더 안, 홈 폴더 자체는 거부한다(홈은 ``allow_home=True`` 일 때만).
- 숨김 폴더/파일, 심볼릭 링크(따라가지 않음), 빌드·캐시·자격증명 폴더(.ssh .aws node_modules .git ...), 비밀번호/키 파일명은 건너뛰고 *이유별 개수와 예시* 를 보고한다.
- 정확히 같은 바이트의 중복은 가장 최근 수정본 하나만 남긴다. 크기가 같은 파일끼리만 해시를 계산해 전체를 다 읽지 않는다.
- 파일 목록 지문(manifest)을 저장해 다음 실행 때 추가/수정/삭제만 알려 준다.
- 이 모듈은 읽기 전용이다. 파일을 만들거나 옮기거나 지우지 않는다(manifest 저장 제외).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .loaders import SafeDocumentLoader, UnsafePathError

DENY_DIR_NAMES = frozenset({
    ".git", ".svn", ".hg", "node_modules", "__pycache__", ".cache", "cache", "dist", "build", "target", ".autorag", ".jikji",
    ".venv", "venv", "site-packages", ".idea", ".vscode", ".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker", ".config",
    ".local", "$recycle.bin", "system volume information", "appdata", "library", "windows", "program files",
    "program files (x86)", "programdata", ".trash", "snap",
})
_DENY_BY_OS = {   # 운영체제별 시스템 영역(소문자, 슬래시). 다른 OS 의 경로(예: macOS 의 /private/var/folders 임시폴더)를 잘못 막지 않도록 나눴다.
    "linux": ("/proc", "/sys", "/dev", "/etc", "/boot", "/bin", "/sbin", "/usr", "/lib", "/lib64", "/run", "/snap", "/var/lib", "/var/log", "/root/.ssh"),
    "darwin": ("/system", "/applications", "/library", "/usr", "/bin", "/sbin", "/etc", "/private/etc", "/private/var/db"),
    "win32": ("c:/windows", "c:/program files", "c:/program files (x86)", "c:/programdata"),
}
SECRET_NAME_PATTERNS = (
    ".env*", "id_rsa*", "id_ed25519*", "id_ecdsa*", "*.pem", "*.key", "*.pfx", "*.p12", "*.kdbx", "*.keystore", "*.jks",
    "credentials*", "secret*", "*password*", "*passwd*", "*token*", "*비밀번호*", "*패스워드*", "*암호*", "*주민등록*",
)
CANDIDATE_FOLDERS = ("Documents", "Downloads", "Desktop", "Notes", "Obsidian", "OneDrive")


class RootRejected(ValueError):
    """위험하거나 모호한 루트."""


@dataclass(frozen=True)
class FileEntry:
    path: Path
    size: int
    mtime_ns: int
    root: Path


@dataclass
class ScanResult:
    roots: list[Path]
    entries: list[FileEntry] = field(default_factory=list)
    skipped: Counter = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    truncated: bool = False

    def note(self, reason: str, path: Path | str) -> None:
        self.skipped[reason] += 1
        if len(self.examples[reason]) < 3:
            self.examples[reason].append(str(path))


def _is_under(p: Path, parent: Path) -> bool:
    return p == parent or parent in p.parents


def validate_root(root: Path, allow_home: bool = False) -> Path:
    r = Path(root).expanduser()
    if not r.is_dir():
        raise RootRejected(f"폴더가 아닙니다: {root}")
    r = r.resolve()
    if r == Path(r.anchor) or len(r.parts) <= 1:
        raise RootRejected(f"파일시스템 루트는 색인할 수 없습니다: {r}")
    if r == Path.home().resolve() and not allow_home:
        raise RootRejected(f"홈 폴더 전체는 승인 없이 색인할 수 없습니다(Documents 같은 하위 폴더를 지정하거나 allow_home=True): {r}")
    import sys

    rs = str(r).replace("\\", "/").lower().rstrip("/")
    for d in _DENY_BY_OS.get(sys.platform if sys.platform in _DENY_BY_OS else "linux", ()):
        if rs == d or rs.startswith(d + "/"):
            raise RootRejected(f"시스템 폴더 안은 색인할 수 없습니다: {r}")
    parts = {x.lower() for x in r.parts[1:]}
    bad = parts & {"node_modules", ".git", ".ssh", ".gnupg", ".aws", "site-packages"}
    if bad:
        raise RootRejected(f"위험한 폴더({sorted(bad)[0]}) 안은 색인할 수 없습니다: {r}")
    return r


def _secret_name(name: str) -> bool:
    low = name.lower()
    return any(fnmatch.fnmatch(low, pat.lower()) for pat in SECRET_NAME_PATTERNS)


def scan_roots(roots: Sequence[Path], extensions: Iterable[str], max_file_bytes: int, *, allow_home: bool = False,
               exclude_paths: Iterable[Path] = (), extra_deny_dirs: Iterable[str] = (), max_files: int = 500_000) -> ScanResult:
    exts = {e.lower() for e in extensions}
    deny_dirs = DENY_DIR_NAMES | {d.lower() for d in extra_deny_dirs}
    excluded = [Path(p).expanduser().resolve() for p in exclude_paths]
    valid_roots: list[Path] = []
    for r in roots:
        v = validate_root(r, allow_home)
        if not any(_is_under(v, o) for o in valid_roots):   # 다른 루트 안에 든 루트는 중복 순회하지 않는다
            valid_roots = [o for o in valid_roots if not _is_under(o, v)] + [v]
    res = ScanResult(valid_roots)
    seen: set[Path] = set()
    for root in valid_roots:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False, onerror=lambda e: res.note("읽기 오류(폴더)", getattr(e, "filename", "?"))):
            here = Path(dirpath)
            keep = []
            for d in sorted(dirnames):
                full = here / d
                if full.is_symlink():
                    res.note("심볼릭 링크 폴더", full)
                elif d.startswith(".") or d.lower() in deny_dirs:
                    res.note("제외 폴더(숨김/시스템/캐시/자격증명)", full)
                elif any(_is_under(full.resolve(), o) for o in excluded):
                    res.note("색인·캐시 저장 폴더", full)
                else:
                    keep.append(d)
            dirnames[:] = keep
            for fn in sorted(filenames):
                p = here / fn
                if Path(fn).suffix.lower() not in exts:
                    res.skipped["지원하지 않는 확장자"] += 1
                    continue
                if fn.lower().endswith(".expect.txt"):
                    res.skipped["정답 메모(.expect.txt)"] += 1; continue
                if fn.startswith((".", "~$")):
                    res.note("숨김/임시 파일", p); continue
                if p.is_symlink():
                    res.note("심볼릭 링크 파일", p); continue
                if _secret_name(fn):
                    res.note("비밀번호·키로 보이는 파일명", p); continue
                try:
                    st = p.stat()
                except OSError:
                    res.note("읽기 오류(파일)", p); continue
                if st.st_size == 0:
                    res.note("빈 파일", p); continue
                if st.st_size > max_file_bytes:
                    res.note("크기 제한 초과", p); continue
                rp = p.resolve()
                if rp in seen:
                    continue
                seen.add(rp)
                res.entries.append(FileEntry(rp, st.st_size, st.st_mtime_ns, root))
                if len(res.entries) >= max_files:
                    res.truncated = True
                    return res
    res.entries.sort(key=lambda e: str(e.path))
    return res


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dedupe_exact(entries: Sequence[FileEntry]) -> tuple[list[FileEntry], list[tuple[FileEntry, FileEntry]]]:
    """바이트까지 같은 파일 중 가장 최근 수정본만 남긴다(동률이면 짧은 경로, 그다음 사전순). (남긴 목록, [(버린 것, 대신 남긴 것)])."""
    by_size: dict[int, list[FileEntry]] = defaultdict(list)
    for e in entries:
        by_size[e.size].append(e)
    kept: list[FileEntry] = []
    dropped: list[tuple[FileEntry, FileEntry]] = []
    for group in by_size.values():
        if len(group) == 1:
            kept.extend(group); continue
        by_hash: dict[str, list[FileEntry]] = defaultdict(list)
        for e in group:
            try:
                by_hash[_sha256_file(e.path)].append(e)
            except OSError:
                kept.append(e)   # 읽을 수 없으면 중복으로 단정하지 않는다
        for same in by_hash.values():
            same.sort(key=lambda e: (-e.mtime_ns, len(str(e.path)), str(e.path)))
            kept.append(same[0])
            dropped.extend((d, same[0]) for d in same[1:])
    kept.sort(key=lambda e: str(e.path))
    return kept, dropped


def build_manifest(entries: Iterable[FileEntry]) -> dict[str, dict]:
    return {str(e.path): {"size": e.size, "mtime_ns": e.mtime_ns} for e in entries}


def diff_manifest(old: dict[str, dict], new: dict[str, dict]) -> dict[str, list[str]]:
    return {"added": sorted(set(new) - set(old)), "removed": sorted(set(old) - set(new)),
            "modified": sorted(p for p in set(old) & set(new) if old[p] != new[p])}


def load_manifest(path: Path) -> dict[str, dict]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_manifest(path: Path, manifest: dict[str, dict]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def suggest_candidate_roots(extensions: Iterable[str], home: Path | None = None, probe_limit: int = 20_000) -> list[dict]:
    """홈 아래의 문서 폴더 후보와 지원 파일 개수(근사). 색인하지 않고 개수만 센다. 사용자가 승인한 것만 루트로 쓴다."""
    exts = {e.lower() for e in extensions}
    base = (home or Path.home()).resolve()
    out = []
    for name in CANDIDATE_FOLDERS:
        d = base / name
        if not d.is_dir() or d.is_symlink():
            continue
        count, seen, truncated = 0, 0, False
        for dirpath, dirnames, filenames in os.walk(d, followlinks=False):
            dirnames[:] = [x for x in dirnames if not x.startswith(".") and x.lower() not in DENY_DIR_NAMES and not (Path(dirpath) / x).is_symlink()]
            for fn in filenames:
                seen += 1
                count += Path(fn).suffix.lower() in exts and not fn.startswith((".", "~$"))
            if seen >= probe_limit:
                truncated = True
                break
        out.append({"path": str(d), "supported_files": count, "truncated": truncated})
    return out


class MultiRootLoader:
    """승인된 여러 루트를 하나의 로더처럼 쓴다. 각 루트는 기존 SafeDocumentLoader 검사(심볼릭 링크·루트 이탈·확장자·크기)를 그대로 받는다."""

    def __init__(self, roots: Sequence[Path], allowed_extensions: Sequence[str], max_file_bytes: int, *, exclude_exact_duplicates: bool = True,
                 allow_home: bool = False, exclude_paths: Iterable[Path] = (), extra_deny_dirs: Iterable[str] = ()):
        valid = []
        for r in roots:
            v = validate_root(r, allow_home)
            if v not in valid:
                valid.append(v)
        if not valid:
            raise RootRejected("루트가 없습니다")
        self.roots = valid
        self.allowed_root = valid[0]   # 기존 코드와의 호환(업로드 폴더 등)
        self.allowed_extensions = tuple(allowed_extensions)
        self.max_file_bytes = max_file_bytes
        self.exclude_exact_duplicates = exclude_exact_duplicates
        self._allow_home, self._exclude, self._deny = allow_home, list(exclude_paths), list(extra_deny_dirs)
        self._loaders = {r: SafeDocumentLoader(r, self.allowed_extensions, max_file_bytes) for r in valid}
        self.last_scan: ScanResult | None = None
        self.last_duplicates: list[tuple[FileEntry, FileEntry]] = []

    def _loader_for(self, path: Path) -> SafeDocumentLoader:
        p = Path(path).expanduser()
        if p.is_symlink():
            raise UnsafePathError(f"symlink rejected: {p}")
        rp = p.resolve(strict=True)
        owners = [r for r in self.roots if _is_under(rp, r)]
        if not owners:
            raise UnsafePathError(f"outside allowed roots: {rp}")
        return self._loaders[max(owners, key=lambda r: len(r.parts))]

    def _validate(self, path: Path) -> Path:
        return self._loader_for(path)._validate(path)

    def load(self, path: Path):
        return self._loader_for(path).load(path)

    def discover(self) -> list[Path]:
        scan = scan_roots(self.roots, self.allowed_extensions, self.max_file_bytes, allow_home=self._allow_home,
                          exclude_paths=self._exclude, extra_deny_dirs=self._deny)
        entries, self.last_duplicates = (dedupe_exact(scan.entries) if self.exclude_exact_duplicates else (scan.entries, []))
        self.last_scan = scan
        return [e.path for e in entries]

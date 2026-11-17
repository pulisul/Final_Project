"""Streamlit 화면용 '폴더에서 문서 찾기' 로직 (화면과 분리한 순수 함수).

원칙
- 폴더 안전 검사는 ``corpus_scan.validate_root`` 가 맡는다(홈 전체·시스템 폴더·.git/.ssh 등 거부). 여기서 새로 만들지 않는다.
- 스캔은 *색인하지 않고* 개수·확장자·제외 사유만 보여 준다. 사용자가 확인한 폴더만 등록한다.
- 등록한 폴더는 ``<index_dir 의 부모>/ui_roots.json`` 에 저장하고, 실행 때마다 다시 검증한다(폴더가 사라졌거나 위험해졌으면 조용히 건너뛰고 알린다).
- 색인은 기존 ``feature_flags.search_roots`` → ``MultiRootLoader`` 경로를 그대로 쓴다.

한계: 등록 해제는 지원하지 않는다. 색인에 들어간 조각을 지우는 기능(삭제 파일 정리)이 아직 없어서, 해제만 하면 옛 조각이 계속 검색된다.
"""
from __future__ import annotations

import dataclasses
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .corpus_scan import RootRejected, dedupe_exact, scan_roots, suggest_candidate_roots, validate_root


def roots_file(cfg) -> Path:
    return Path(cfg.index_dir).parent / "ui_roots.json"


def load_roots(cfg) -> tuple[list[Path], list[str]]:
    """저장된 폴더 목록 → (아직 유효한 폴더, 건너뛴 사유). 파일이 없거나 깨졌으면 빈 목록."""
    p = roots_file(cfg)
    if not p.exists():
        return [], []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        items = [str(x) for x in raw["roots"]] if isinstance(raw, dict) else []
    except Exception as exc:  # noqa: BLE001 - 깨진 파일은 무시하고 알린다
        return [], [f"{p.name} 을 읽을 수 없음: {type(exc).__name__}"]
    ok, notes = [], []
    for s in items:
        try:
            r = validate_root(Path(s), allow_home=False)
        except RootRejected as exc:
            notes.append(f"건너뜀: {exc}")
            continue
        if r not in ok:
            ok.append(r)
    return ok, notes


def save_roots(cfg, roots: list[Path]) -> None:
    p = roots_file(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"roots": [str(r) for r in roots]}, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def check_root(cfg, text: str) -> Path:
    """입력 문자열 → 검증된 절대 경로. 위험하거나 폴더가 아니면 RootRejected.
    색인·증거 저장 폴더 안은 거부한다(자기 자신을 색인하는 루프 방지). 그 폴더를 포함하는 상위 폴더는 허용하되 스캔·색인에서 자동 제외된다."""
    s = (text or "").strip().strip('"').strip("'")
    if not s:
        raise RootRejected("폴더 경로를 입력하세요")
    r = validate_root(Path(s), allow_home=False)
    for own in (Path(cfg.index_dir), Path(cfg.evidence_dir)):
        own = own.resolve()
        if r == own or own in r.parents:
            raise RootRejected(f"색인·증거 저장 폴더와 겹칩니다: {r}")
    return r


def add_root(cfg, text: str) -> tuple[Path, bool]:
    """폴더를 검증해 등록한다. (경로, 새로 추가됐는지)."""
    r = check_root(cfg, text)
    roots, _ = load_roots(cfg)
    if r in roots:
        return r, False
    save_roots(cfg, [*roots, r])
    return r, True


def apply_roots(cfg, roots):
    """cfg 사본에 search_roots 를 넣는다. 원본 cfg 와 feature_flags 는 바꾸지 않는다."""
    flags = dict(cfg.feature_flags or {})
    if roots:
        flags["search_roots"] = [str(r) for r in roots]
    return dataclasses.replace(cfg, feature_flags=flags)


@dataclass
class FolderScan:
    root: Path
    files: int = 0
    unique_files: int = 0
    duplicates: int = 0
    total_mb: float = 0.0
    by_ext: dict = field(default_factory=dict)
    skipped: dict = field(default_factory=dict)
    examples: dict = field(default_factory=dict)
    samples: list = field(default_factory=list)
    truncated: bool = False


def scan_folder(cfg, root: Path, *, sample: int = 15, max_files: int = 50_000) -> FolderScan:
    """색인 없이 훑어서 요약한다. 색인에 쓸 것과 같은 규칙(scan_roots)을 쓴다."""
    own = [Path(cfg.index_dir), Path(cfg.evidence_dir), Path(cfg.index_dir).parent / "parse_cache"]
    res = scan_roots([root], cfg.allowed_extensions, cfg.max_file_bytes, exclude_paths=own, max_files=max_files)
    uniq, dups = dedupe_exact(res.entries)
    base = res.roots[0] if res.roots else root
    ext = Counter(e.path.suffix.lower() for e in uniq)
    out = FolderScan(
        root=base, files=len(res.entries), unique_files=len(uniq), duplicates=len(dups),
        total_mb=round(sum(e.size for e in uniq) / (1024 * 1024), 2),
        by_ext=dict(ext.most_common()), skipped=dict(res.skipped), examples={k: list(v) for k, v in res.examples.items()},
        truncated=res.truncated,
    )
    for e in uniq[:sample]:
        try:
            out.samples.append(str(e.path.relative_to(base)))
        except ValueError:
            out.samples.append(str(e.path))
    return out


def suggest_folders(cfg) -> list[dict]:
    """홈 아래 흔한 문서 폴더(Documents 등)와 지원 파일 개수. 개수만 세고 색인하지 않는다."""
    return [c for c in suggest_candidate_roots(cfg.allowed_extensions) if c["supported_files"] > 0]


def registered_documents(cfg) -> list[Path]:
    """현재 설정(기본 문서 폴더 + 등록 폴더)에서 색인 대상이 되는 파일 목록."""
    from .document_ingest import make_loader

    return list(make_loader(cfg).discover())

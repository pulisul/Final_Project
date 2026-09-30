from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import IntEnum
from hashlib import sha256
from pathlib import Path
from typing import Iterable
import json

class Severity(IntEnum):
    BLOCKER = 0
    CRITICAL = 1
    MAJOR = 2
    MINOR = 3
    INFO = 4

@dataclass(frozen=True)
class Issue:
    issue_id: str
    title: str
    severity: Severity
    component: str
    reproducible: bool = True
    workaround: str | None = None
    status: str = "open"

    @property
    def blocks_release(self) -> bool:
        return self.status != "closed" and self.severity <= Severity.CRITICAL


def classify_issue(title: str, component: str, *, data_loss=False, security=False,
                   core_unusable=False, no_workaround=False, cosmetic=False) -> Severity:
    if data_loss or security or core_unusable:
        return Severity.BLOCKER
    if no_workaround:
        return Severity.CRITICAL
    if cosmetic:
        return Severity.MINOR
    return Severity.MAJOR


def release_blockers(issues: Iterable[Issue]) -> list[Issue]:
    return [i for i in issues if i.blocks_release]


def file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_freeze_manifest(root: Path, files: Iterable[Path], label: str="P0-RC1") -> dict:
    rows = {}
    for p in sorted({Path(x).resolve() for x in files}):
        rel = str(p.relative_to(root.resolve()))
        rows[rel] = file_sha256(p)
    return {"label": label, "files": rows}


def verify_freeze_manifest(root: Path, manifest: dict) -> dict:
    changed, missing = [], []
    for rel, expected in manifest.get("files", {}).items():
        p = root / rel
        if not p.exists():
            missing.append(rel)
        elif file_sha256(p) != expected:
            changed.append(rel)
    return {"pass": not changed and not missing, "changed": changed, "missing": missing}


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def issue_to_dict(issue: Issue) -> dict:
    d = asdict(issue)
    d["severity"] = issue.severity.name
    d["blocks_release"] = issue.blocks_release
    return d

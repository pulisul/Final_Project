from __future__ import annotations
import hashlib, json, platform, sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

EXCLUDE_PARTS={'.git','.pytest_cache','__pycache__','.venv'}
EXCLUDE_SUFFIXES={'.pyc','.pyo'}
DYNAMIC_RELEASE_PATHS={
    'evidence/week9/RELEASE_MANIFEST.json',
    'evidence/week9/RELEASE_MANIFEST_VERIFY.json',
    'evidence/week9/WEEK9_GATE.json',
    'evidence/week9/WEEK9_STRICT_GATE.json',
}


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def release_files(root: Path) -> list[Path]:
    out=[]
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        rel=p.relative_to(root)
        if any(part in EXCLUDE_PARTS for part in rel.parts):
            continue
        if p.suffix in EXCLUDE_SUFFIXES:
            continue
        if rel.as_posix() in DYNAMIC_RELEASE_PATHS:
            continue
        if rel.parts and rel.parts[0]=='index':
            continue
        out.append(p)
    return sorted(out)


def build_release_manifest(root: Path, label: str, files: Iterable[Path]|None=None) -> dict:
    files=list(files) if files is not None else release_files(root)
    entries=[]
    for p in files:
        rel=p.relative_to(root).as_posix()
        entries.append({'path':rel,'bytes':p.stat().st_size,'sha256':sha256_file(p)})
    return {
        'schema_version':1,
        'release_label':label,
        'python_runtime':platform.python_version(),
        'python_required':'3.11.x',
        'file_count':len(entries),
        'files':entries,
    }


def verify_release_manifest(root: Path, manifest: dict) -> dict:
    changed=[]; missing=[]
    for item in manifest.get('files',[]):
        p=root/item['path']
        if not p.exists():
            missing.append(item['path']); continue
        if sha256_file(p)!=item['sha256']:
            changed.append(item['path'])
    return {'pass':not changed and not missing,'changed':changed,'missing':missing,'checked':len(manifest.get('files',[]))}


def validate_submission_bundle(root: Path) -> dict:
    required=[
        'README.md','ARCHITECTURE.md','requirements.txt','requirements-dev.txt','pyproject.toml',
        'docs/final/ARCHITECTURE_FINAL.md','docs/final/ERD_FINAL.md','docs/final/RL_FLOW_FINAL.md',
        'docs/final/EVALUATION_REPORT.md','docs/final/DATA_ANALYSIS_REPORT.md',
        'docs/final/DEMO_SCRIPT.md','docs/final/SUBMISSION_CHECKLIST.md',
        'evidence/week8/WEEK8_GATE.json','evidence/week7/WEEK7_GATE.json',
    ]
    missing=[r for r in required if not (root/r).exists()]
    return {'pass':not missing,'required':required,'missing':missing}

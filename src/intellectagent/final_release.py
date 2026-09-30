from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def critical_paths(root: Path) -> list[Path]:
    candidates = [
        root / "config/core.json",
        root / "config/core.production.json",
        root / "config/release_policy.json",
        root / "requirements.txt",
        root / "requirements-dev.txt",
        root / "requirements-lock.txt",
        root / "artifacts/week7/active_policy.json",
    ]
    src = root / "src/intellectagent"
    if src.exists():
        candidates.extend(sorted(p for p in src.rglob("*.py") if "__pycache__" not in p.parts))
    return [p for p in candidates if p.exists()]


def snapshot_paths(root: Path, paths: Iterable[Path] | None = None) -> dict[str, str]:
    selected = list(paths) if paths is not None else critical_paths(root)
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(selected)}


def compare_snapshots(before: dict[str, str], after: dict[str, str]) -> dict[str, Any]:
    all_keys = sorted(set(before) | set(after))
    added = [k for k in all_keys if k not in before]
    missing = [k for k in all_keys if k not in after]
    changed = [k for k in all_keys if k in before and k in after and before[k] != after[k]]
    return {
        "pass": not added and not missing and not changed,
        "added": added,
        "missing": missing,
        "changed": changed,
        "count_before": len(before),
        "count_after": len(after),
    }


def gate_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"pass": False, "exists": False, "path": str(path)}
    try:
        data = load_json(path)
    except Exception as exc:
        return {"pass": False, "exists": True, "path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    passed = bool(data.get("overall_pass", data.get("pass", False)))
    return {"pass": passed, "exists": True, "path": str(path), "sha256": sha256_file(path), "data": data}


def active_policy_summary(root: Path) -> dict[str, Any]:
    path = root / "artifacts/week7/active_policy.json"
    if not path.exists():
        return {"pass": False, "error": "active_policy_missing"}
    data = load_json(path)
    name = data.get("active_policy")
    allowed = {"random", "rule", "linucb"}
    return {
        "pass": name in allowed,
        "active_policy": name,
        "rollback_applied": bool(data.get("rollback_applied", False)),
        "reason": data.get("reason"),
        "sha256": sha256_file(path),
    }


def collect_metrics(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    candidate = root / "evidence/strict/18_candidate_benchmark.json"
    if candidate.exists():
        d = load_json(candidate)
        out["candidate_benchmark"] = {
            "pass": bool(d.get("pass", False)),
            "selected_candidate_k": d.get("selected_candidate_k"),
            "embedding_model": d.get("embedding_model"),
            "reranker_model": d.get("reranker_model"),
            "results": [
                {"candidate_k": r.get("candidate_k"), **r.get("metrics", {})}
                for r in d.get("results", [])
            ],
            "gpu_before": d.get("gpu_before"),
            "gpu_after": d.get("gpu_after"),
        }
    ragas = root / "evidence/strict/23_ragas_strict.json"
    if ragas.exists():
        d = load_json(ragas)
        out["ragas"] = {"pass": bool(d.get("pass", False)), "smoke2": d.get("smoke2"), "full20": d.get("full20")}
    ollama = root / "evidence/strict/21_ollama_grounded.json"
    if ollama.exists():
        d = load_json(ollama)
        out["ollama_grounded"] = {k: d.get(k) for k in ("pass", "model", "citation_present", "grounding_pass", "elapsed_ms") if k in d}
    gpu = root / "evidence/strict/14_gpu_runtime.json"
    if gpu.exists():
        out["gpu_runtime"] = load_json(gpu)
    return out


def release_evidence_paths(root: Path) -> list[Path]:
    rels = [
        "evidence/strict/00_preflight.json",
        "evidence/strict/STEP6_9_GATE.json",
        "evidence/strict/STEP10_13_GATE.json",
        "evidence/strict/STEP14_INTEGRATED_GATE.json",
        "evidence/strict/30_production_e2e.json",
        "evidence/week9/WEEK9_GATE.json",
        "artifacts/week7/active_policy.json",
        "requirements-lock.txt",
    ]
    return [root / r for r in rels if (root / r).exists()]


def release_identity(root: Path, evidence_paths: Iterable[Path] | None = None) -> dict[str, Any]:
    paths = list(evidence_paths) if evidence_paths is not None else release_evidence_paths(root)
    hashes = {str(p.relative_to(root)): sha256_file(p) for p in sorted(paths)}
    canonical = json.dumps(hashes, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    release_id = hashlib.sha256(canonical).hexdigest()
    return {"release_id": release_id, "files": hashes}

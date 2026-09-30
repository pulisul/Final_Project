from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import socket
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Any


@dataclass
class StrictCheck:
    name: str
    passed: bool
    required: bool = True
    details: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["pass"] = d.pop("passed")
        return d


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def tcp_open(host: str, port: int, timeout: float = 1.0) -> tuple[bool, str | None]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, None
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"


def environment_snapshot(root: Path) -> dict[str, Any]:
    lock = root / "requirements-lock.txt"
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": platform.platform(),
        "requirements_lock_exists": lock.exists(),
        "requirements_lock_sha256": sha256_file(lock) if lock.exists() else None,
        "packages": {
            n: package_available(n)
            for n in (
                "sentence_transformers",
                "chromadb",
                "ollama",
                "langgraph",
                "ragas",
                "streamlit",
            )
        },
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def run_check(name: str, fn: Callable[[], dict[str, Any]], required: bool = True) -> StrictCheck:
    try:
        details = fn()
        passed = bool(details.get("pass", False))
        return StrictCheck(name=name, passed=passed, required=required, details=details)
    except Exception as exc:
        return StrictCheck(name=name, passed=False, required=required, error=f"{type(exc).__name__}: {exc}")


def summarize(checks: list[StrictCheck]) -> dict[str, Any]:
    required = [c for c in checks if c.required]
    return {
        "overall_pass": all(c.passed for c in required),
        "required_passed": sum(1 for c in required if c.passed),
        "required_total": len(required),
        "checks": [c.to_dict() for c in checks],
    }

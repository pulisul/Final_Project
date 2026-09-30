from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def build_ragas_records(raw_log_path: Path) -> list[dict[str, Any]]:
    records = []
    for line in raw_log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        records.append({
            "user_input": row["question"],
            "response": row["answer"],
            "retrieved_contexts": row.get("retrieved_contexts", []),
            "reference": row.get("reference_answer", ""),
        })
    return records


def require_ragas() -> Any:
    try:
        import ragas
        return ragas
    except ImportError as e:
        raise RuntimeError("ragas>=0.4,<0.5 is not installed; deterministic Week 6 metrics remain available") from e


def ragas_import_smoke() -> dict[str, Any]:
    try:
        import ragas
        from ragas.dataset_schema import SingleTurnSample  # noqa: F401
        from ragas.metrics import Faithfulness  # noqa: F401
        return {"pass": True, "ragas_version": getattr(ragas, "__version__", "unknown")}
    except Exception as e:
        return {"pass": False, "error": f"{type(e).__name__}: {e}"}


def run_with_injected_evaluator(records: list[dict[str, Any]], evaluator: Callable[[list[dict[str, Any]]], dict[str, Any]]) -> dict[str, Any]:
    """Stable project boundary around RAGAS.

    Production code can inject a version-specific/local-LLM evaluator after the
    2-case smoke test. Core evaluation never imports RAGAS directly.
    """
    return evaluator(records)

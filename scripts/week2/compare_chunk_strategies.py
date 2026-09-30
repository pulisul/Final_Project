from __future__ import annotations
import json
from pathlib import Path


def main() -> int:
    base = Path("evidence/week2")
    s = json.loads((base / "semantic_result.json").read_text(encoding="utf-8"))
    l = json.loads((base / "length_result.json").read_text(encoding="utf-8"))
    sm, lm = s["metrics"], l["metrics"]
    # Primary: retrieval quality; tie-breaker: lower mean latency; never hide a loss.
    if (sm["hit_rate_at_k"], sm["mrr"]) > (lm["hit_rate_at_k"], lm["mrr"]):
        winner = "semantic"
    elif (sm["hit_rate_at_k"], sm["mrr"]) < (lm["hit_rate_at_k"], lm["mrr"]):
        winner = "length"
    else:
        winner = "semantic" if sm["latency_ms_mean"] <= lm["latency_ms_mean"] else "length"
    report = {
        "semantic": sm,
        "length": lm,
        "winner": winner,
        "decision_rule": "maximize hit_rate_at_k, then mrr; if tied, minimize mean latency",
        "warning": "sample-doc smoke results are not production evidence; rerun on real corpus/eval15 before promotion",
    }
    out = base / "chunk_strategy_comparison.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

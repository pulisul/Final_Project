from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

REQ = [
    Path("data/eval/week2/eval15.jsonl"),
    Path("evidence/week2/semantic_config.json"),
    Path("evidence/week2/semantic_result.json"),
    Path("evidence/week2/length_config.json"),
    Path("evidence/week2/length_result.json"),
    Path("evidence/week2/chunk_strategy_comparison.json"),
]


def main() -> int:
    checks = {}
    p = subprocess.run([sys.executable, "-m", "pytest", "-q"], text=True, capture_output=True)
    checks["pytest"] = {"pass": p.returncode == 0, "detail": (p.stdout + p.stderr)[-2000:]}
    for path in REQ:
        checks[str(path)] = {"pass": path.exists(), "detail": "exists" if path.exists() else "missing"}

    if Path("evidence/week2/chunk_strategy_comparison.json").exists():
        comp = json.loads(Path("evidence/week2/chunk_strategy_comparison.json").read_text(encoding="utf-8"))
        winner_metrics = comp[comp["winner"]]
        checks["retrieval_quality"] = {
            "pass": winner_metrics["hit_rate_at_k"] >= 0.70 and winner_metrics["mrr"] >= 0.60,
            "detail": winner_metrics,
        }
    else:
        checks["retrieval_quality"] = {"pass": False, "detail": "comparison missing"}

    overall = all(v["pass"] for v in checks.values())
    report = {"week": 2, "overall_pass": overall, "checks": checks}
    out = Path("evidence/week2/WEEK2_GATE.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if overall else 1

if __name__ == "__main__":
    raise SystemExit(main())

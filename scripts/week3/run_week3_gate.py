from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict-bge", action="store_true", help="Require real BGE reranker smoke + benchmark before promotion")
    args = ap.parse_args()
    checks = {}
    p = subprocess.run([sys.executable, "-m", "pytest", "-q"], text=True, capture_output=True)
    checks["pytest"] = {"pass": p.returncode == 0, "detail": (p.stdout + p.stderr)[-2000:]}

    required = [
        Path("evidence/week3/none_c10_k5.json"),
        Path("evidence/week3/token-overlap_c10_k5.json"),
        Path("evidence/week3/retrieval_comparison.json"),
        Path("evidence/week3/candidate_k_tuning.json"),
    ]
    for path in required:
        checks[str(path)] = {"pass": path.exists(), "detail": "exists" if path.exists() else "missing"}

    if Path("evidence/week3/token-overlap_c10_k5.json").exists():
        r = json.loads(Path("evidence/week3/token-overlap_c10_k5.json").read_text(encoding="utf-8"))
        m = r["metrics"]
        checks["retrieval_quality"] = {
            "pass": m["hit_rate_at_k"] >= 0.70 and m["mrr"] >= 0.60,
            "detail": m,
        }
        checks["reranker_fallback_smoke"] = {
            "pass": m["reranker_fallback_rate"] == 0.0,
            "detail": m["reranker_fallback_rate"],
        }
    else:
        checks["retrieval_quality"] = {"pass": False, "detail": "missing"}
        checks["reranker_fallback_smoke"] = {"pass": False, "detail": "missing"}

    if args.strict_bge:
        smoke = Path("evidence/week3/bge_reranker_smoke.json")
        bench = Path("evidence/week3/bge_c10_k5.json")
        smoke_data = json.loads(smoke.read_text(encoding="utf-8")) if smoke.exists() else {}
        checks["bge_reranker_smoke"] = {"pass": bool(smoke_data.get("pass")), "detail": smoke_data or "missing"}
        checks["bge_benchmark"] = {"pass": bench.exists(), "detail": "exists" if bench.exists() else "missing"}

    overall = all(v["pass"] for v in checks.values())
    report = {"week": 3, "mode": "strict-bge" if args.strict_bge else "smoke", "overall_pass": overall, "checks": checks}
    out = Path("evidence/week3/WEEK3_STRICT_GATE.json" if args.strict_bge else "evidence/week3/WEEK3_GATE.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())

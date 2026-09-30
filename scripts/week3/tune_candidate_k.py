from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

CANDIDATES = (5, 10, 20)


def main() -> int:
    rows = []
    for candidate_k in CANDIDATES:
        out = Path(f"evidence/week3/tune_c{candidate_k}_k5.json")
        cmd = [
            sys.executable,
            "scripts/week3/run_retrieval_benchmark.py",
            "--reranker", "token-overlap",
            "--candidate-k", str(candidate_k),
            "--top-k", "5",
            "--out", str(out),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = "src" + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        p = subprocess.run(cmd, text=True, capture_output=True, env=env)
        if p.returncode != 0:
            print(p.stdout)
            print(p.stderr, file=sys.stderr)
            return p.returncode
        data = json.loads(out.read_text(encoding="utf-8"))
        rows.append({"candidate_k": candidate_k, **data["metrics"]})

    # Predeclared rule: maximize hit rate, then MRR, then choose lower latency and smaller candidate_k.
    best = sorted(
        rows,
        key=lambda r: (-r["hit_rate_at_k"], -r["mrr"], r["latency_ms_mean"], r["candidate_k"]),
    )[0]
    report = {
        "candidate_space": list(CANDIDATES),
        "selection_rule": "max Hit Rate@k -> max MRR -> min mean latency -> min candidate_k",
        "results": rows,
        "selected_candidate_k": best["candidate_k"],
        "warning": "This smoke tuning uses token-overlap reranking. Re-run the same 5/10/20 space with real BGE before production promotion.",
    }
    path = Path("evidence/week3/candidate_k_tuning.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

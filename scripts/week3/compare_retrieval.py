from __future__ import annotations

import json
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    base = load("evidence/week3/none_c10_k5.json")
    rerank = load("evidence/week3/token-overlap_c10_k5.json")
    bm = base["metrics"]
    rm = rerank["metrics"]
    # Predeclared selection rule: quality first; latency is tie-break only.
    if (rm["hit_rate_at_k"], rm["mrr"]) > (bm["hit_rate_at_k"], bm["mrr"]):
        winner = "rerank"
    elif (rm["hit_rate_at_k"], rm["mrr"]) < (bm["hit_rate_at_k"], bm["mrr"]):
        winner = "baseline"
    else:
        winner = "rerank" if rm["latency_ms_mean"] <= bm["latency_ms_mean"] * 1.50 else "baseline"
    report = {
        "selection_rule": ["hit_rate_at_k", "mrr", "latency <= baseline*1.50 on quality tie"],
        "baseline": bm,
        "rerank": rm,
        "winner": winner,
        "quality_delta": {
            "hit_rate_at_k": rm["hit_rate_at_k"] - bm["hit_rate_at_k"],
            "mrr": rm["mrr"] - bm["mrr"],
            "latency_ms_mean": rm["latency_ms_mean"] - bm["latency_ms_mean"],
        },
        "warning": "token-overlap is a deterministic smoke reranker; production promotion requires bge_reranker_smoke.json plus the same benchmark with --reranker bge.",
    }
    out = Path("evidence/week3/retrieval_comparison.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

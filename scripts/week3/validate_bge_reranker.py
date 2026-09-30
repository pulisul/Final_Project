from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--out", default="evidence/week3/bge_reranker_smoke.json")
    args = ap.parse_args()

    report = {"model": args.model, "pass": False, "error": None}
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(args.model)
        pairs = [
            ("RAG란 무엇인가요?", "RAG는 검색과 생성 모델을 결합하는 방식입니다."),
            ("RAG란 무엇인가요?", "사과는 과일입니다."),
            ("사과는 무엇인가요?", "사과는 과일입니다."),
        ]
        t0 = time.perf_counter()
        scores = [float(x) for x in model.predict(pairs)]
        elapsed = (time.perf_counter() - t0) * 1000
        finite = all(math.isfinite(x) for x in scores)
        ranking_ok = scores[0] > scores[1]
        report.update({
            "scores": scores,
            "latency_ms": elapsed,
            "finite": finite,
            "ranking_smoke_pass": ranking_ok,
            "pass": finite and ranking_ok,
        })
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

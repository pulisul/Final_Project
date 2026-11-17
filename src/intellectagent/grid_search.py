"""격자 탐색 + 최적 선택 (AutoRAG ``node`` 평가와 ``strategy.select_best`` 의 핵심 아이디어).

AutoRAG 방식: 한 노드(예: 검색)의 후보 모듈들을 같은 질문셋으로 돌려 지표를 계산하고,
(1) 속도 임계값을 넘는 후보를 거르되 모두 걸러지면 전부 유지하고(``avoid_empty_result``),
(2) ``mean``(지표 평균) 또는 ``rank``(지표별 순위의 역수 평균) 전략으로 하나를 고른다.
선택 결과와 전 후보의 표를 파일로 남겨 "무엇을 바꿨더니 얼마나 달라졌나"의 근거로 쓴다.
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .models import EvalCase, SearchHit
from .retrieval_contents import CONTENT_METRICS
from .retrieval_metrics import METRICS, case_to_gt, score_case

SearchFn = Callable[[str, int], list[SearchHit]]

DEFAULT_METRICS = ("retrieval_f1", "retrieval_recall", "retrieval_precision",
                   "retrieval_ndcg", "retrieval_map", "retrieval_mrr")


@dataclass
class Candidate:
    name: str
    search: SearchFn
    params: dict = field(default_factory=dict)


def evaluate_candidate(cand: Candidate, cases: Sequence[EvalCase], top_k: int,
                       metrics: Sequence[str] = DEFAULT_METRICS) -> dict:
    per_case: dict[str, list[float]] = {m: [] for m in metrics}
    lat: list[float] = []
    for case in cases:
        t = time.perf_counter()
        hits = cand.search(case.question, top_k)
        lat.append((time.perf_counter() - t) * 1000)
        ids = [h.doc_id for h in hits]  # 문서 단위 평가(기존 hit_rate_at_k 와 같은 단위). 같은 문서의 중복 청크는 한 번만.
        ids = list(dict.fromkeys(ids))
        s = score_case(case_to_gt(case), ids, [m for m in metrics if m in METRICS])
        for m in metrics:
            if m in CONTENT_METRICS:  # 청크 텍스트 vs 정답 근거 텍스트: 청킹 설정이 달라도 비교 가능
                gt_contents = getattr(case, "gt_contents", None)
                if not gt_contents:
                    raise ValueError(f"metric {m} needs case.gt_contents (case {getattr(case, 'case_id', '?')} has none)")
                per_case[m].append(CONTENT_METRICS[m](gt_contents, [h.text for h in hits]))
            else:
                per_case[m].append(s[m])
    row = {"candidate": cand.name, **{f"param_{k}": v for k, v in cand.params.items()}, "top_k": top_k, "n": len(cases)}
    for m in metrics:
        row[m] = statistics.mean(per_case[m]) if per_case[m] else 0.0
    row["latency_ms_mean"] = statistics.mean(lat) if lat else 0.0
    return row


def filter_by_speed(rows: list[dict], speed_threshold_ms: float | None) -> list[dict]:
    if speed_threshold_ms is None:
        return rows
    kept = [r for r in rows if r["latency_ms_mean"] <= speed_threshold_ms]
    return kept or rows  # AutoRAG avoid_empty_result: 전부 걸러지면 원본 유지


def select_best(rows: list[dict], metrics: Sequence[str] = DEFAULT_METRICS, strategy: str = "mean") -> dict:
    if not rows:
        raise ValueError("no rows to select from")
    if strategy == "mean":
        return max(rows, key=lambda r: statistics.mean(r[m] for m in metrics))
    if strategy == "rank":
        # AutoRAG select_best_rr: 지표별 평균을 내림차순 순위(pandas rank 기본값 = 동점은 *평균 순위*)로 바꿔 1/순위를 합산.
        score = {id(r): 0.0 for r in rows}
        for m in metrics:
            for r in rows:
                greater = sum(1 for o in rows if o[m] > r[m])
                equal = sum(1 for o in rows if o[m] == r[m])
                score[id(r)] += 1.0 / (1 + greater + (equal - 1) / 2)
        return max(rows, key=lambda r: score[id(r)])
    if strategy == "normalize_mean":
        # AutoRAG select_normalize_mean: 지표별 min-max 정규화 합. 모든 후보가 같은 값인 지표(max==min)는 NaN 이 되어 합에서 빠진다(=0).
        lo = {m: min(r[m] for r in rows) for m in metrics}
        hi = {m: max(r[m] for r in rows) for m in metrics}
        def total(r):
            return round(sum((r[m] - lo[m]) / (hi[m] - lo[m]) for m in metrics if hi[m] != lo[m]), 12)  # 부동소수 잡음으로 동점이 뒤집히지 않게
        return max(rows, key=total)
    raise ValueError(f"unknown strategy {strategy!r} (mean|rank|normalize_mean)")


def run_grid(candidates: Sequence[Candidate], cases: Sequence[EvalCase], top_k: int = 5,
             metrics: Sequence[str] = DEFAULT_METRICS, strategy: str = "mean",
             speed_threshold_ms: float | None = None, out_dir: Path | None = None) -> dict:
    unknown = [m for m in metrics if m not in METRICS and m not in CONTENT_METRICS]
    if unknown:
        raise KeyError(f"unknown metrics: {unknown}")
    rows = [evaluate_candidate(c, cases, top_k, metrics) for c in candidates]
    eligible = filter_by_speed(rows, speed_threshold_ms)
    best = select_best(eligible, metrics, strategy)
    result = {"strategy": strategy, "metrics": list(metrics), "top_k": top_k,
              "speed_threshold_ms": speed_threshold_ms, "best": best["candidate"], "rows": rows}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "grid_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "candidate", k))
        with (out_dir / "grid_result.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return result

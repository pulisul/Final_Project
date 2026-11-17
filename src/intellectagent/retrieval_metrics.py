"""검색 지표 (AutoRAG legacy `evaluation/metric/retrieval.py` 의 정의를 따른다).

정답 표기: ``gt_groups`` 는 "그룹의 리스트"다. 각 그룹은 *그중 하나만 찾아도* 그 그룹을 맞힌 것으로 본다.
  - ``[["d1"], ["d2"]]``  -> d1 과 d2 를 각각 찾아야 함 (recall 분모 = 2)
  - ``[["d1", "d1b"]]``   -> d1 또는 d1b 중 하나면 충분 (같은 내용의 중복 문서 등)

우리 ``EvalCase.relevant_doc_ids`` 는 ``case_to_gt`` 로 "문서마다 별도 그룹" 으로 바꾼다.
(기존 hit_rate_at_k 는 "하나라도 맞히면 1" 이므로 그룹 하나짜리와 같다 — 두 정의를 함께 보고한다.)

원본과 동일한 결과인지는 ``scripts/autorag/crosscheck.py`` 로 원본 함수와 직접 대조한다.
"""

from __future__ import annotations

import itertools
import math
from typing import Callable, Iterable, Sequence

from .models import EvalCase

GtGroups = Sequence[Sequence[str]]


def case_to_gt(case: EvalCase) -> list[list[str]]:
    return [[d] for d in case.relevant_doc_ids]


def retrieval_recall(gt: GtGroups, pred: Sequence[str]) -> float:
    gt_sets = [frozenset(g) for g in gt]
    pred_set = set(pred)
    hits = sum(any(p in s for p in pred_set) for s in gt_sets)
    return hits / len(gt) if len(gt) > 0 else 0.0


def retrieval_precision(gt: GtGroups, pred: Sequence[str]) -> float:
    gt_sets = [frozenset(g) for g in gt]
    pred_set = set(pred)
    hits = sum(any(p in s for s in gt_sets) for p in pred_set)
    return hits / len(pred) if len(pred) > 0 else 0.0


def retrieval_f1(gt: GtGroups, pred: Sequence[str]) -> float:
    r, p = retrieval_recall(gt, pred), retrieval_precision(gt, pred)
    return 0.0 if r + p == 0 else 2 * r * p / (r + p)


def retrieval_ndcg(gt: GtGroups, pred: Sequence[str]) -> float:
    gt_sets = [frozenset(g) for g in gt]
    rel = {p: 1 if any(p in s for s in gt_sets) else 0 for p in set(pred)}
    dcg = sum((2 ** rel[d] - 1) / math.log2(i + 2) for i, d in enumerate(pred))
    n_gt = len(list(itertools.chain.from_iterable(gt)))
    ideal = [1] * min(n_gt, len(pred)) + [0] * max(0, len(pred) - n_gt)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def retrieval_mrr(gt: GtGroups, pred: Sequence[str]) -> float:
    gt_sets = [frozenset(g) for g in gt]
    rr = []
    for s in gt_sets:
        for i, p in enumerate(pred):
            if p in s:
                rr.append(1.0 / (i + 1))
                break
    return sum(rr) / len(gt_sets) if rr else 0.0


def retrieval_map(gt: GtGroups, pred: Sequence[str]) -> float:
    gt_sets = [frozenset(g) for g in gt]
    aps = []
    for s in gt_sets:
        hits = [1 if p in s else 0 for p in pred]
        precs = [sum(hits[: i + 1]) / (i + 1) for i, h in enumerate(hits) if h == 1]
        aps.append(sum(precs) / len(precs) if precs else 0.0)
    return sum(aps) / len(gt_sets) if aps else 0.0


METRICS: dict[str, Callable[[GtGroups, Sequence[str]], float]] = {
    "retrieval_f1": retrieval_f1,
    "retrieval_recall": retrieval_recall,
    "retrieval_precision": retrieval_precision,
    "retrieval_ndcg": retrieval_ndcg,
    "retrieval_map": retrieval_map,
    "retrieval_mrr": retrieval_mrr,
}


def score_case(gt: GtGroups, pred_ids: Sequence[str], metrics: Iterable[str] | None = None) -> dict[str, float]:
    names = list(metrics) if metrics is not None else list(METRICS)
    unknown = [m for m in names if m not in METRICS]
    if unknown:
        raise KeyError(f"unknown metrics: {unknown}")
    return {m: METRICS[m](gt, pred_ids) for m in names}

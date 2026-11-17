"""실환경(실제 임베딩·Ollama·실문서) 평가용 순수 계산 함수 모음.

스크립트(``scripts/autorag/real_env_eval.py``)는 모델·파일을 다루고, 여기서는 숫자만 다룬다.
그래서 모델 없이도 테스트할 수 있다. 판정 기준은 *제안값*이다(추정): 팀이 합의해 인자로 바꿀 수 있다.

용어
- hit@k   : 정답 문서가 상위 k 문서 안에 있으면 1.
- MRR     : 정답 문서가 처음 나온 순위의 역수(없으면 0). 같은 문서의 중복 청크는 한 번으로 센다.
- gate    : ``CoreRAG._gate(hits)`` 값. 이 값이 ``fallback_min_score`` 미만이면 "근거 없음" 으로 답을 막는다.
- 오차단율(false_fallback_rate) : 근거가 있는 질문인데 gate 때문에 막힌 비율. 높으면 쓸모없는 챗봇이 된다.
- 정거절률(correct_refusal_rate): 근거가 없는 질문을 gate 가 막은 비율. 낮으면 지어낸 답이 나간다.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence


# ---- 순위 지표 ------------------------------------------------------------------------------------------------------
def first_relevant_rank(doc_ids: Sequence[str], relevant: Sequence[str]) -> int | None:
    """중복을 제거한 문서 순서에서 정답 문서가 처음 나오는 순위(1부터). 없으면 None."""
    rel = set(relevant)
    seen: list[str] = []
    for d in doc_ids:
        if d not in seen:
            seen.append(d)
    for i, d in enumerate(seen, start=1):
        if d in rel:
            return i
    return None


def rank_metrics(ranks: Sequence[int | None], k: int) -> dict[str, float]:
    """질문별 순위 → hit@k, MRR@k. k 밖의 순위는 miss(0)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    n = len(ranks)
    if n == 0:
        return {"hit": 0.0, "mrr": 0.0, "n": 0}
    hit = sum(1 for r in ranks if r is not None and r <= k) / n
    mrr = sum(1.0 / r for r in ranks if r is not None and r <= k) / n
    return {"hit": hit, "mrr": mrr, "n": n}


def bootstrap_ci(values: Sequence[float], iterations: int = 2000, seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    """평균의 부트스트랩 신뢰구간(백분위법). 값이 없으면 (0,0), 하나뿐이면 (값,값)."""
    vals = list(values)
    if not vals:
        return (0.0, 0.0)
    if len(vals) == 1:
        return (vals[0], vals[0])
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(iterations))
    lo = means[int((alpha / 2) * (iterations - 1))]
    hi = means[int((1 - alpha / 2) * (iterations - 1))]
    return (lo, hi)


# ---- gate(fallback) 분석 ----------------------------------------------------------------------------------------------
def auc(pos: Sequence[float], neg: Sequence[float]) -> float | None:
    """pos(근거 있는 질문의 gate) 가 neg(근거 없는 질문의 gate) 보다 클 확률. 동점은 0.5. 한쪽이 비면 None."""
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else 0.5 if p == q else 0.0
    return wins / (len(pos) * len(neg))


@dataclass(frozen=True)
class GatePoint:
    threshold: float
    false_fallback_rate: float   # 근거 있는 질문 중 막힌 비율 (gate < threshold)
    correct_refusal_rate: float  # 근거 없는 질문 중 막힌 비율
    balanced_accuracy: float


def gate_point(answerable: Sequence[float], unanswerable: Sequence[float], threshold: float) -> GatePoint:
    ff = sum(1 for g in answerable if g < threshold) / len(answerable) if answerable else 0.0
    cr = sum(1 for g in unanswerable if g < threshold) / len(unanswerable) if unanswerable else 0.0
    bal = ((1 - ff) + cr) / 2 if answerable and unanswerable else (1 - ff if answerable else cr)
    return GatePoint(threshold, ff, cr, bal)


def gate_sweep(answerable: Sequence[float], unanswerable: Sequence[float], max_false_fallback: float = 0.10) -> dict:
    """가능한 모든 임계값(관측된 gate 값의 중간점)을 훑어 추천 임계값을 낸다.

    추천 규칙: 오차단율 ≤ max_false_fallback 인 점 중 정거절률이 가장 높은 점(동점이면 임계값이 낮은 쪽 = 덜 막는 쪽).
    그런 점이 없으면 추천값은 None (점수만으로는 근거 있음/없음을 구분하지 못한다).
    """
    scores = sorted({float(g) for g in (*answerable, *unanswerable) if g is not None and not math.isnan(g)})
    if not scores:
        return {"points": [], "recommended": None, "auc": None}
    cands = [scores[0] - 1e-9] + [(a + b) / 2 for a, b in zip(scores, scores[1:])] + [scores[-1] + 1e-9]
    points = [gate_point(answerable, unanswerable, t) for t in cands]
    ok = [p for p in points if p.false_fallback_rate <= max_false_fallback]
    rec = max(ok, key=lambda p: (p.correct_refusal_rate, -p.threshold)) if ok and unanswerable else (min(ok, key=lambda p: p.threshold) if ok else None)
    return {"points": points, "recommended": rec, "auc": auc(answerable, unanswerable)}


# ---- 판정 ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Criteria:
    min_hit: float = 0.80
    min_mrr: float = 0.60
    max_false_fallback: float = 0.10
    min_correct_refusal: float = 0.80
    min_questions: int = 30
    min_unanswerable: int = 5


def verdict(metrics: Mapping[str, float], gate: GatePoint | None, n_answerable: int, n_unanswerable: int,
            crit: Criteria = Criteria(), valid_embedder: bool = True) -> dict:
    """항목별 통과/실패와 종합. 실제 임베더가 아니면 종합은 항상 INVALID (배관 확인용)."""
    checks = {
        "hit": {"value": metrics.get("hit", 0.0), "min": crit.min_hit, "pass": metrics.get("hit", 0.0) >= crit.min_hit},
        "mrr": {"value": metrics.get("mrr", 0.0), "min": crit.min_mrr, "pass": metrics.get("mrr", 0.0) >= crit.min_mrr},
    }
    if gate is not None and n_unanswerable:
        checks["false_fallback"] = {"value": gate.false_fallback_rate, "max": crit.max_false_fallback,
                                    "pass": gate.false_fallback_rate <= crit.max_false_fallback}
        checks["correct_refusal"] = {"value": gate.correct_refusal_rate, "min": crit.min_correct_refusal,
                                     "pass": gate.correct_refusal_rate >= crit.min_correct_refusal}
    warnings = []
    if n_answerable < crit.min_questions:
        warnings.append(f"답 있는 질문이 {n_answerable}개(<{crit.min_questions}): 수치가 우연일 수 있다.")
    if n_unanswerable < crit.min_unanswerable:
        warnings.append(f"답 없는 질문이 {n_unanswerable}개(<{crit.min_unanswerable}): 거절 성능을 믿기 어렵다.")
    if not valid_embedder:
        overall = "INVALID"
        warnings.append("HashEmbedder 로 돌렸다: 배관 확인용이며 성능 증거가 아니다.")
    else:
        overall = "PASS" if all(c["pass"] for c in checks.values()) else "FAIL"
    return {"overall": overall, "checks": checks, "warnings": warnings}


def percentile(values: Sequence[float], p: float) -> float:
    """선형 보간 백분위(p=0..100)."""
    if not values:
        return 0.0
    v = sorted(values)
    if len(v) == 1:
        return v[0]
    pos = (len(v) - 1) * p / 100
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


# ---- 층별 보고 · 실패 유형 · Pareto · 점검 목록 ------------------------------------------------------------------------------
def overlap_band(x: float, hi: float = 0.8, mid: float = 0.5) -> str:
    """질문-청크 어휘 겹침(lexical_overlap) 층. high 층이 점수를 끌어올리고 있지 않은지 보려는 것(청크 낱말을 베낀 질문은 BM25 에 유리)."""
    return "high" if x >= hi else "mid" if x >= mid else "low"


def stratify(items: Sequence[tuple[str, int | None]], k: int) -> dict[str, dict[str, float]]:
    """(층 이름, 순위) 목록 → 층별 {n, hit, mrr}. 한 층에 질문이 적으면 수치가 흔들리니 n 을 함께 본다."""
    groups: dict[str, list[int | None]] = {}
    for key, rank in items:
        groups.setdefault(key, []).append(rank)
    return {key: rank_metrics(ranks, k) for key, ranks in sorted(groups.items())}


OUTCOMES = ("ok", "refused", "retrieval_miss", "wrong_source", "answer_mismatch", "correct_refusal", "hallucinated_answer")


def classify_outcome(*, expected_fallback: bool, fallback: bool, gold_in_hits: bool, gold_cited: bool,
                     answer_overlap: float | None, min_overlap: float = 0.3) -> str:
    """질문 하나의 종단 결과를 실패 유형으로 분류한다(Agentic-AutoRAG 의 failure_mode 를 우리 상황에 맞게 줄여 재구현).

    답 없는 질문: correct_refusal(막음) / hallucinated_answer(답을 내놓음 → 가장 위험).
    답 있는 질문:
      refused         검색은 정답 문서를 가져왔는데 gate 나 근거 검증이 막았다(너무 엄격한 보호)
      retrieval_miss  정답 문서를 못 가져왔다(그 결과 거절됐든 엉뚱하게 답했든)
      wrong_source    답했지만 정답 문서를 인용하지 않았다
      answer_mismatch 정답 문서를 인용했지만 답이 기준 답과 크게 다르다(어휘 기반 추정 — 사람이 확인)
      ok              위에 해당 없음
    """
    if expected_fallback:
        return "correct_refusal" if fallback else "hallucinated_answer"
    if not gold_in_hits:
        return "retrieval_miss"
    if fallback:
        return "refused"
    if not gold_cited:
        return "wrong_source"
    if answer_overlap is not None and answer_overlap < min_overlap:
        return "answer_mismatch"
    return "ok"


def pareto_frontier(points: Sequence[tuple[str, float, float]]) -> set[str]:
    """(이름, 점수↑, 지연시간↓) 중 어느 것에도 지배당하지 않는 이름들. 점수·지연이 모두 같은 둘은 서로 지배하지 않는다."""
    out: set[str] = set()
    for name, s, t in points:
        dominated = any((s2 >= s and t2 <= t) and (s2 > s or t2 < t) for n2, s2, t2 in points if n2 != name)
        if not dominated:
            out.add(name)
    return out


def never_found(ranks_by_candidate: Mapping[str, Mapping[str, int | None]], k: int) -> list[str]:
    """모든 후보가 top-k 안에서 정답 문서를 못 찾은 질문 id. 라벨 오류·문서 파싱 누락·너무 모호한 질문일 가능성이 높아 먼저 사람이 본다."""
    cands = list(ranks_by_candidate.values())
    if not cands:
        return []
    ids = sorted(set().union(*[set(c) for c in cands]))
    return [i for i in ids if all(c.get(i) is None or c[i] > k for c in cands)]

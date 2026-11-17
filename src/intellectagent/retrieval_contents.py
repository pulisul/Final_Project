"""내용 기반 검색 지표 (AutoRAG legacy ``evaluation/metric/retrieval_contents.py``).

청킹 방식을 바꾸면 청크 ID 가 달라져 ID 기반 지표(recall/ndcg…)로는 비교할 수 없다.
이 지표는 *검색된 청크의 텍스트* 를 *정답 근거 텍스트* 와 토큰 단위로 비교하므로 청킹과 무관하게 비교된다.

정의 (원본과 동일): 검색된 각 청크마다 모든 정답 텍스트와의 (precision, recall, f1) 중 최댓값을 취하고,
그 값을 검색된 청크 전체에 대해 평균한다. 토큰화는 SQuAD 정규화(소문자, ASCII 구두점/영문 관사 제거, 공백 분리).

해석 주의
- ``token_recall`` 은 "정답 근거가 검색 결과에 들어 있는가"(청크가 클수록 유리).
- ``token_precision`` 은 "검색된 청크가 얼마나 근거만 담고 있는가"(청크가 클수록 불리).
  그래서 청크 크기가 다른 설정끼리는 f1 하나로 우열을 가리지 말고 recall 과 precision 을 함께 본다.
- 기본 토큰은 *공백 어절* 이라 한국어 조사 변이를 흡수하지 못한다(원본 동일). ``tokenizer=`` 로 문자 bigram 등을 줄 수 있다.
- 검색 결과나 정답이 비어 있으면 0.0 (원본은 예외).
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Callable, Iterable, Sequence, Union

Tokenizer = Callable[[str], list]
Contents = Union[Sequence[str], Sequence[Sequence[str]]]

_EXCLUDE = set(string.punctuation)


def normalize_string(s: str) -> str:
    """SQuAD v1.1 정규화 (원본 ``autorag.utils.util.normalize_string`` 과 같은 순서)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _EXCLUDE)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _default_tokens(s: str) -> list:
    return normalize_string(s).split()


def single_token_f1(ground_truth: str, prediction: str, tokenizer: Tokenizer | None = None) -> tuple[float, float, float]:
    tok = tokenizer or _default_tokens
    pred, gt = tok(prediction), tok(ground_truth)
    same = sum((Counter(pred) & Counter(gt)).values())
    if same == 0:
        return 0.0, 0.0, 0.0
    precision, recall = same / len(pred), same / len(gt)
    return precision, recall, 2 * precision * recall / (precision + recall)


def _flatten(gt: Contents) -> list[str]:
    out: list[str] = []
    for item in gt:
        if isinstance(item, str):
            out.append(item)
        else:
            out.extend(_flatten(item))
    return out


def _score(idx: int, gt: Contents, pred: Iterable[str], tokenizer: Tokenizer | None) -> float:
    gts, preds = _flatten(gt), list(pred)
    if not gts or not preds:
        return 0.0
    best = [max(single_token_f1(g, p, tokenizer)[idx] for g in gts) for p in preds]
    return sum(best) / len(best)


def retrieval_token_precision(gt: Contents, pred: Iterable[str], tokenizer: Tokenizer | None = None) -> float:
    return _score(0, gt, pred, tokenizer)


def retrieval_token_recall(gt: Contents, pred: Iterable[str], tokenizer: Tokenizer | None = None) -> float:
    return _score(1, gt, pred, tokenizer)


def retrieval_token_f1(gt: Contents, pred: Iterable[str], tokenizer: Tokenizer | None = None) -> float:
    return _score(2, gt, pred, tokenizer)


CONTENT_METRICS: dict[str, Callable[..., float]] = {
    "retrieval_token_f1": retrieval_token_f1,
    "retrieval_token_recall": retrieval_token_recall,
    "retrieval_token_precision": retrieval_token_precision,
}

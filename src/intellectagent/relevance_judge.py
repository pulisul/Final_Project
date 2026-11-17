"""청크별 관련성 판정 (선택 기능, 기본 꺼짐).

아이디어 출처(재구현, 코드 복사 아님): MAIN-RAG(ACL 2025)의 판정 에이전트와 적응형 임계값 `평균 − n·표준편차`,
CRAG 의 "맞음/모호/틀림" 구분, Self-RAG 의 IsRel. 논문들은 학습된 모델이나 로그확률을 쓰지만 여기서는 로컬 LLM 에게
"이 문서가 질문에 답할 정보를 담고 있나? Yes/No" 를 묻는 프롬프트 근사다. 논문 성능이 이 구현에 그대로 적용된다고 볼 수 없다.

고정 ``fallback_min_score`` 는 임베더·검색 방식마다 다시 보정해야 한다. 이 판정은 질문마다 청크를 직접 읽어
(1) 절대 기준(Yes 가 No 보다 우세)에 못 미치는 청크를 버리고, (2) 남은 청크 중 점수가 평균 − n·표준편차 미만인 것을 더 버린다. n 의 기본값 1.0 은 필요한 근거를 버리지 않으려는 보수적 선택이다(MAIN-RAG 원문 기본은 0).
모두 버려지면 빈 결과 → CoreRAG 가 기존 방식 그대로 거절한다.

안전 원칙: 이 판정은 *강화* 기능이다. LLM 이 죽거나 이상한 답을 하면 원래 검색 결과를 그대로 돌려준다(fail-open):
판정기가 없다고 기존 보호가 약해지지는 않는다(기존 gate 와 grounding 은 그대로 적용). 단, 이 경우 ``last_error`` 에 남는다.
호출 비용: 질문당 최대 ``max_candidates`` 번 LLM 호출 — 지연시간은 ``real_env_eval.py --judge`` 로 측정한 뒤 켠다.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from statistics import pstdev, mean
from typing import Any, Callable, Sequence

from .models import SearchHit

JUDGE_PROMPT = (
    "다음 [문서]가 [질문]에 답하는 데 필요한 정보를 직접 담고 있는지 판단하세요. "
    "관련 주제이지만 답이 되는 정보가 없으면 No 입니다. 반드시 Yes 또는 No 한 단어로만 답하세요.\n\n"
    "[질문]\n{question}\n\n[문서]\n{document}\n\n답:"
)
_THINK = re.compile(r"<think>.*?</think>", re.S)


@dataclass(frozen=True)
class JudgeReply:
    text: str
    yes_logprob: float | None = None
    no_logprob: float | None = None


Judge = Callable[[str], JudgeReply]


def score_reply(reply: JudgeReply) -> float | None:
    """Yes/No 응답 → 점수. 양쪽 로그확률이 있으면 log P(Yes) − log P(No), 없으면 텍스트로 +1/−1. 판독 불가면 None."""
    if reply.yes_logprob is not None and reply.no_logprob is not None:
        return float(reply.yes_logprob - reply.no_logprob)
    t = _THINK.sub("", reply.text or "").strip().lower().lstrip("\"'`*[( ")
    if t.startswith("yes") or t.startswith("예") or t.startswith("네"):
        return 1.0
    if t.startswith("no") or t.startswith("아니"):
        return -1.0
    return None


def adaptive_threshold(scores: Sequence[float], n_std: float = 0.0) -> float:
    """평균 − n·표준편차 (모표준편차). 점수가 하나뿐이면 그 값 자체."""
    if not scores:
        return -math.inf
    return mean(scores) - n_std * (pstdev(scores) if len(scores) > 1 else 0.0)


class OllamaJudge:
    """Ollama 로 Yes/No 를 묻는다. 로그확률은 지원되면(Ollama ≥ 0.12.11, 네이티브 API) 쓰고, 아니면 텍스트로 대체한다.

    ⚠ 로그확률 응답 구조는 실제 Ollama 로 확인하지 못했다(추정): ``logprobs`` 목록의 첫 토큰 항목에서 ``top_logprobs`` 의
    (token, logprob) 쌍을 찾는다. 구조가 달라 파싱이 실패하면 조용히 텍스트 방식으로 돌아간다.
    """

    def __init__(self, model: str, client: Any | None = None, use_logprobs: bool = True):
        self.model, self._client, self.use_logprobs = model, client, use_logprobs

    def _cli(self):
        if self._client is not None:
            return self._client
        import ollama
        return ollama

    @staticmethod
    def _get(obj: Any, key: str, default=None):
        return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

    def _extract(self, resp: Any) -> JudgeReply:
        msg = self._get(resp, "message")
        text = str(self._get(msg, "content", "") or "") if msg is not None else str(self._get(resp, "response", "") or "")
        yes = no = None
        try:
            lp = self._get(resp, "logprobs")
            first = lp[0] if lp else None
            tops = self._get(first, "top_logprobs") or []
            cand = [(str(self._get(t, "token", "")).strip().lower(), self._get(t, "logprob")) for t in tops]
            cand.append((str(self._get(first, "token", "")).strip().lower(), self._get(first, "logprob")))
            ys = [v for k, v in cand if k in ("yes", "예") and v is not None]
            ns = [v for k, v in cand if k in ("no", "아니") and v is not None]
            if ys and ns:
                yes, no = max(ys), max(ns)
        except Exception:  # noqa: BLE001 - 구조가 다르면 텍스트로
            yes = no = None
        return JudgeReply(text, yes, no)

    def __call__(self, prompt: str) -> JudgeReply:
        cli = self._cli()
        msgs = [{"role": "user", "content": prompt}]
        opts = {"temperature": 0, "num_predict": 4}
        if self.use_logprobs:
            try:
                return self._extract(cli.chat(model=self.model, messages=msgs, options=opts, logprobs=True, top_logprobs=5))
            except TypeError:       # 구버전 클라이언트는 logprobs 인자를 모른다
                self.use_logprobs = False
        return self._extract(cli.chat(model=self.model, messages=msgs, options=opts))


class RelevanceJudge:
    def __init__(self, judge: Judge, *, n_std: float = 1.0, min_score: float = 0.0, max_candidates: int = 5, max_chars: int = 1500):
        """min_score: 절대 기준. 점수가 이 값 '이하'면 버린다(0.0 = Yes 가 No 보다 우세해야 통과)."""
        self.judge, self.n_std, self.min_score = judge, float(n_std), float(min_score)
        self.max_candidates, self.max_chars = int(max_candidates), int(max_chars)
        self.last_error: str | None = None
        self.last_scores: list[float | None] = []

    def filter(self, question: str, hits: Sequence[SearchHit]) -> list[SearchHit]:
        self.last_error = None; self.last_scores = []
        hits = list(hits)
        if not hits:
            return []
        head = hits[: self.max_candidates]
        scores: list[float | None] = []
        for h in head:
            try:
                scores.append(score_reply(self.judge(JUDGE_PROMPT.format(question=question, document=h.text[: self.max_chars]))))
            except Exception as exc:  # noqa: BLE001 - 판정기 장애 = 강화 기능 실패, 기존 보호는 그대로
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.last_scores = scores
                return hits
        self.last_scores = scores
        valid = [s for s in scores if s is not None]
        if not valid:                                   # 한 줄도 판독하지 못함 → 판정 불가, 원본 유지
            self.last_error = "unparseable judge replies"
            return hits
        positives = [s for s in valid if s > self.min_score]
        thr = adaptive_threshold(positives, self.n_std) if positives else math.inf
        # 판정하지 않은 꼬리(max_candidates 밖)는 근거로 쓰지 않는다: 검증 안 된 청크가 보호를 우회하면 안 된다.
        return [SearchHit(h.chunk_id, h.doc_id, h.text, h.score, {**h.metadata, "judge_score": s})
                for h, s in zip(head, scores) if s is not None and s > self.min_score and s >= thr]


class JudgedRetriever:
    """어떤 검색기든 감싸서 결과를 관련성 판정으로 거른다. 모두 걸러지면 빈 목록을 돌려준다."""

    def __init__(self, inner: Any, judge: RelevanceJudge):
        self.inner, self.judge = inner, judge

    def search(self, query: str, top_k: int, *args, **kwargs) -> list[SearchHit]:
        hits = list(self.inner.search(query, top_k, *args, **kwargs))
        return self.judge.filter(query, hits)[:top_k]

import math

import pytest

from intellectagent.models import SearchHit
from intellectagent.pipeline import CoreRAG
from intellectagent.relevance_judge import (JudgedRetriever, JudgeReply, OllamaJudge, RelevanceJudge, adaptive_threshold, score_reply)

A, B, C = "a" * 20, "b" * 20, "c" * 20


def H(cid, text, score=0.9, **m):
    return SearchHit(cid, "d" + cid[0], text, score, m)


class Scripted:
    """문서 본문에 들어 있는 표식으로 점수를 정하는 가짜 판정기."""
    def __init__(self, table, fail=False): self.table, self.fail, self.calls = table, fail, 0
    def __call__(self, prompt):
        self.calls += 1
        if self.fail: raise RuntimeError("ollama down")
        for k, v in self.table.items():
            if k in prompt: return v
        return JudgeReply("???")


# ---- 점수화 ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text,exp", [("Yes", 1.0), (" yes.", 1.0), ("No", -1.0), ("no, because", -1.0), ('"Yes"', 1.0), ("예", 1.0), ("아니오", -1.0),
                                       ("<think>음...</think>Yes", 1.0), ("maybe", None), ("", None), ("Yesterday I", 1.0)])
def test_score_reply_text(text, exp):
    # "Yesterday" 는 접두어 규칙 때문에 Yes 로 읽힌다: 한 단어로 답하라고 요구하고 num_predict 를 4 로 제한해 위험을 줄인다(알려진 한계).
    assert score_reply(JudgeReply(text)) == exp


def test_score_reply_prefers_logprobs():
    assert score_reply(JudgeReply("No", yes_logprob=-0.1, no_logprob=-2.1)) == pytest.approx(2.0)   # 텍스트와 달라도 로그확률 우선
    assert score_reply(JudgeReply("Yes", yes_logprob=-0.1, no_logprob=None)) == 1.0                  # 한쪽만 있으면 텍스트


def test_adaptive_threshold():
    assert adaptive_threshold([1, 2, 3], 0) == 2 and adaptive_threshold([1, 2, 3], 1) == pytest.approx(2 - (2 / 3) ** 0.5)
    assert adaptive_threshold([5], 3) == 5 and adaptive_threshold([], 1) == -math.inf


# ---- 필터 --------------------------------------------------------------------------------------------------------------
YES, NO = JudgeReply("Yes"), JudgeReply("No")


def test_filter_drops_no_and_keeps_yes_with_score_metadata():
    hits = [H(A, "DOC-A 연차 15일"), H(B, "DOC-B 주차장 안내"), H(C, "DOC-C 병가 60일")]
    j = RelevanceJudge(Scripted({"DOC-A": YES, "DOC-B": NO, "DOC-C": YES}))
    out = j.filter("연차는?", hits)
    assert [h.chunk_id for h in out] == [A, C] and all(h.metadata["judge_score"] == 1.0 for h in out)
    assert j.last_error is None and j.last_scores == [1.0, -1.0, 1.0]


def test_all_no_gives_empty_so_core_falls_back():
    hits = [H(A, "DOC-B 주차장 안내")]
    r = JudgedRetriever(type("R", (), {"search": lambda s, q, k, candidate_k=None: hits})(), RelevanceJudge(Scripted({"DOC-B": NO})))
    class G:
        def generate(self, q, h): raise AssertionError("거절이어야 하므로 생성기를 부르면 안 됨")
    a = CoreRAG(r, G(), 5, 0.12).ask("연차는?")
    assert a.fallback and not a.citations


def test_logprob_scores_apply_adaptive_threshold():
    hits = [H(A, "x1"), H(B, "x2"), H(C, "x3")]
    table = {"x1": JudgeReply("Yes", -0.1, -4.0), "x2": JudgeReply("Yes", -0.5, -0.9), "x3": JudgeReply("Yes", -0.2, -3.9)}   # 3.9, 0.4, 3.7
    strict = RelevanceJudge(Scripted(table), n_std=0.0).filter("q", hits)       # 평균 2.667 미만(0.4)은 버림
    loose = RelevanceJudge(Scripted(table), n_std=3.0).filter("q", hits)
    assert [h.chunk_id for h in strict] == [A, C] and [h.chunk_id for h in loose] == [A, B, C]


def test_negative_scores_never_pass_even_if_above_mean():
    hits = [H(A, "x1"), H(B, "x2")]
    j = RelevanceJudge(Scripted({"x1": JudgeReply("No", -3.0, -0.1), "x2": JudgeReply("No", -2.0, -0.1)}), n_std=5)
    assert j.filter("q", hits) == []        # 상대 기준만 쓰면 둘 중 하나는 살아남는다 — 절대 기준이 막는다


def test_judge_failure_is_fail_open_and_recorded():
    hits = [H(A, "x"), H(B, "y")]
    sj = Scripted({}, fail=True); j = RelevanceJudge(sj)
    assert j.filter("q", hits) == hits and "ollama down" in j.last_error and sj.calls == 1       # 첫 실패에서 중단
    j2 = RelevanceJudge(Scripted({}))                                                           # 전부 판독 불가
    assert j2.filter("q", hits) == hits and j2.last_error == "unparseable judge replies"


def test_unreadable_reply_among_valid_ones_is_dropped():
    hits = [H(A, "x1"), H(B, "x2")]
    out = RelevanceJudge(Scripted({"x1": YES, "x2": JudgeReply("흠")})).filter("q", hits)
    assert [h.chunk_id for h in out] == [A]


def test_max_candidates_limits_calls_and_drops_unjudged_tail():
    hits = [H(A, "x1"), H(B, "x2"), H(C, "x3")]
    sj = Scripted({"x": YES}); out = RelevanceJudge(sj, max_candidates=2).filter("q", hits)
    assert sj.calls == 2 and [h.chunk_id for h in out] == [A, B]


def test_empty_hits_make_no_calls_and_gate_score_survives():
    sj = Scripted({}); assert RelevanceJudge(sj).filter("q", []) == [] and sj.calls == 0
    out = RelevanceJudge(Scripted({"x": YES})).filter("q", [H(A, "x", 0.03, gate_score=0.6)])
    assert out[0].metadata["gate_score"] == 0.6


def test_prompt_truncates_document():
    seen = []
    RelevanceJudge(lambda p: (seen.append(p), YES)[1], max_chars=10).filter("질문", [H(A, "가" * 100)])
    assert "가" * 10 in seen[0] and "가" * 11 not in seen[0] and "질문" in seen[0]


# ---- Ollama 어댑터 -----------------------------------------------------------------------------------------------------
class FakeClient:
    def __init__(self, resp, accepts_logprobs=True): self.resp, self.accepts, self.kw = resp, accepts_logprobs, None
    def chat(self, model, messages, options=None, **kw):
        if kw and not self.accepts: raise TypeError("unexpected keyword 'logprobs'")
        self.kw = kw; return self.resp


def test_ollama_judge_parses_assumed_logprob_schema_and_falls_back():
    resp = {"message": {"content": "Yes"}, "logprobs": [{"token": "Yes", "logprob": -0.2, "top_logprobs": [{"token": "Yes", "logprob": -0.2}, {"token": "No", "logprob": -1.7}]}]}
    r = OllamaJudge("m", FakeClient(resp))("p")
    assert r.yes_logprob == -0.2 and r.no_logprob == -1.7 and score_reply(r) == pytest.approx(1.5)
    bad = {"message": {"content": "No"}, "logprobs": "garbage"}
    r2 = OllamaJudge("m", FakeClient(bad))("p")
    assert r2.yes_logprob is None and score_reply(r2) == -1.0                                   # 구조가 달라도 텍스트로
    old = OllamaJudge("m", FakeClient({"message": {"content": "Yes"}}, accepts_logprobs=False))
    assert score_reply(old("p")) == 1.0 and old.use_logprobs is False                           # 구버전 클라이언트


def test_factory_wraps_retriever_only_when_flag_on():
    import dataclasses
    from pathlib import Path
    from intellectagent.config import CoreConfig
    from intellectagent.factory import build_components
    base = CoreConfig.load(Path(__file__).parents[1] / "config/core.json")
    off = build_components(dataclasses.replace(base, feature_flags={}))
    on = build_components(dataclasses.replace(base, feature_flags={"use_relevance_judge": True, "judge_std_multiplier": 2.0}))
    rag = lambda parts: next(v for v in parts if hasattr(v, "retriever"))
    assert not isinstance(rag(off).retriever, JudgedRetriever)
    j = rag(on).retriever
    assert isinstance(j, JudgedRetriever) and j.judge.n_std == 2.0


def test_absolute_floor_holds_even_when_adaptive_threshold_goes_negative():
    hits = [H(A, "x1"), H(B, "x2"), H(C, "x3")]
    table = {"x1": JudgeReply("Yes", -0.1, -10.1), "x2": JudgeReply("Yes", -0.1, -0.2), "x3": JudgeReply("No", -2.0, -0.1)}   # 10.0, 0.1, -1.9
    out = RelevanceJudge(Scripted(table), n_std=5.0).filter("q", hits)
    assert [h.chunk_id for h in out] == [A, B]       # n_std 가 커서 임계값이 음수가 돼도 No(-1.9)는 통과 못 한다

import pytest

from intellectagent.gating import gate_score
from intellectagent.generation import OllamaGenerator, build_grounded_prompt
from intellectagent.grounding import validate_grounding
from intellectagent.models import SearchHit
from intellectagent.orchestration import AgenticRAGWorkflow
from intellectagent.pipeline import CoreRAG

A, B = "a" * 20, "b" * 20


def hit(cid=A, score=0.9, **meta):
    return SearchHit(cid, "d", "연차는 15일이다.", score, meta)


# ---- gate 일치 (워크플로가 gate_score 를 무시하던 결함) ---------------------------------------------------------------------
def test_gate_score_rules():
    assert gate_score([]) == 0.0
    assert gate_score([hit(score=0.03, gate_score=0.62)]) == 0.62
    assert gate_score([hit(score=0.5), hit(B, 0.9)]) == 0.5                      # gate_score 없으면 1순위 점수
    assert gate_score([hit(score=0.7, gate_score=None)]) == 0.7                  # BM25 단독 결과(None)는 1순위 점수로
    assert gate_score([hit(score=0.03, gate_score=0.2), hit(B, 0.02, gate_score=0.4)]) == 0.4


class _R:
    def __init__(self, hits): self.hits = hits
    def search(self, q, k, candidate_k=None): return self.hits


class _G:
    def generate(self, q, h): return f"연차는 15일입니다. [{A}]"


@pytest.mark.parametrize("score,gate,expect_fallback", [(0.03, 0.62, False), (0.03, 0.05, True), (0.9, 0.05, True)])
def test_core_and_workflow_agree_on_fallback(score, gate, expect_fallback):
    hits = [hit(score=score, gate_score=gate)]
    core = CoreRAG(_R(hits), _G(), 5, 0.12).ask("q").fallback
    wf = AgenticRAGWorkflow(_R(hits), _G(), fallback_min_score=0.12).run("q")[0].fallback
    assert core == wf == expect_fallback


# ---- grounding ------------------------------------------------------------------------------------------------------------
def test_label_line_without_citation_is_allowed_but_claim_line_is_not():
    h = [hit()]
    assert validate_grounding(f"답변:\n연차는 15일입니다. [{A}]", h).ok
    assert not validate_grounding(f"답변:\n연차는 15일입니다.", h).ok                       # 인용 없음
    assert not validate_grounding(f"연차는 15일입니다. [{A}]\n병가는 60일입니다.", h).ok    # 주장 줄에 인용 없음
    long_label = "이것은 길이가 스무 자를 넘는 머리말처럼 보이는 문장입니다:"
    assert not validate_grounding(f"{long_label}\n연차는 15일입니다. [{A}]", h).ok          # 긴 줄은 주장으로 취급


def test_numeric_citations_map_to_chunk_ids_in_order():
    h = [hit(A), hit(B)]
    r = validate_grounding("연차는 15일입니다. [1]\n병가는 60일입니다. [2]", h)
    assert r.ok and r.cited_ids == (A, B)
    r = validate_grounding("두 규정을 모두 따릅니다. [2, 1]", h)
    assert r.ok and r.cited_ids == (B, A)


def test_numeric_citation_out_of_range_or_zero_is_rejected():
    h = [hit(A)]
    for t in ("연차는 15일입니다. [2]", "연차는 15일입니다. [0]", "연차는 15일입니다. [1][3]"):
        r = validate_grounding(t, h)
        assert not r.ok and r.unsupported_citations, t


def test_id_style_still_works_and_wrong_id_still_rejected():
    h = [hit()]
    assert validate_grounding(f"연차는 15일입니다. [{A}]", h).ok
    assert not validate_grounding("연차는 15일입니다. [%s]" % ("a" * 19 + "b"), h).ok
    assert not validate_grounding("연차는 15일입니다.", h).ok and not validate_grounding("", h).ok


def test_known_limit_semantic_support_is_not_checked():
    """문서화된 한계: 유효한 인용만 있으면 내용이 틀려도 통과한다(의미 검증은 별도). 이 테스트가 바뀌면 문서도 바꿀 것."""
    assert validate_grounding(f"연차는 30일입니다. [{A}]", [hit()]).ok


# ---- 프롬프트 / 생성기 ----------------------------------------------------------------------------------------------------
def test_prompt_styles():
    h = [hit(A), hit(B)]
    pid = build_grounded_prompt("q", h)
    assert f"SOURCE_ID={A}" in pid and "[1]" not in pid
    pix = build_grounded_prompt("q", h, "index")
    assert "[1]\n연차는" in pix and "[2]\n연차는" in pix and A not in pix and B not in pix
    with pytest.raises(ValueError):
        build_grounded_prompt("q", h, "nope")
    with pytest.raises(ValueError):
        OllamaGenerator("m", citation_style="nope")


def test_ollama_generator_uses_index_style_end_to_end():
    seen = {}
    class Client:
        def chat(self, model, messages):
            seen["p"] = messages[0]["content"]; return {"message": {"content": "연차는 15일입니다. [1]"}}
    hits = [hit()]
    pipe = CoreRAG(_R(hits), OllamaGenerator("m", Client(), citation_style="index"), 5, 0.12)
    a = pipe.ask("연차?")
    assert not a.fallback and a.citations == (A,) and "SOURCE_ID" not in seen["p"]


def test_factory_passes_citation_style_to_generator():
    import dataclasses
    from pathlib import Path
    from intellectagent.config import CoreConfig
    from intellectagent.factory import build_components
    base = CoreConfig.load(Path(__file__).parents[1] / "config/core.json")
    def gen(flags):
        parts = build_components(dataclasses.replace(base, feature_flags=flags))
        seq = parts.values() if isinstance(parts, dict) else parts
        return next(v.generator for v in seq if hasattr(v, "generator") and isinstance(v.generator, OllamaGenerator))
    assert gen({"use_ollama": True, "citation_style": "index"}).citation_style == "index"
    assert gen({"use_ollama": True}).citation_style == "index"  # 기본값은 번호 인용(index)

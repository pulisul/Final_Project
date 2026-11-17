import importlib.util
import json
import sys
from pathlib import Path

import pytest

from intellectagent.loaders import SafeDocumentLoader
from intellectagent.models import SearchHit
from intellectagent.qa_gen import (QARecord, answer_overlap, answers_agree, classify_question, generate_qa, oracle_filter, read_jsonl,
                                   write_jsonl)
from intellectagent.real_eval import (classify_outcome, never_found, overlap_band, pareto_frontier, stratify)

SPEC = importlib.util.spec_from_file_location("ree2", Path(__file__).parents[2] / "scripts/autorag/real_env_eval.py")
ree = importlib.util.module_from_spec(SPEC); sys.modules["ree2"] = ree; SPEC.loader.exec_module(ree)


# ---- 질문 유형 -------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("q,t", [
    ("연차휴가는 며칠 부여되나요?", "numeric"), ("병가 한도는 얼마인가요?", "numeric"),
    ("재택근무와 출근근무의 차이는 무엇인가요?", "comparison"), ("두 제도를 비교하면 어떤가요?", "comparison"),
    ("휴가를 신청하려면 어떻게 하나요?", "procedure"), ("결재 절차는 어떻게 되나요?", "procedure"),
    ("RAG 란 무엇인가요?", "definition"), ("보안 등급의 의미는?", "definition"),
    ("백업은 누가 담당하나요?", "factoid"),
])
def test_classify_question(q, t):
    assert classify_question(q) == t


def test_comparison_wins_over_numeric_and_definition():
    assert classify_question("A와 B의 차이는 무엇이며 비용은 얼마인가요?") == "comparison"


# ---- 답 일치 -------------------------------------------------------------------------------------------------------------
def test_answers_agree_rules():
    assert answers_agree("연차는 15일입니다.", "15일")                    # 포함
    assert answers_agree("매일 새벽 2시에 전체 백업을 수행한다", "매일 새벽 2시에 전체 백업을 수행합니다")  # dice
    assert not answers_agree("연차는 15일입니다.", "병가는 60일까지 가능합니다")
    assert not answers_agree("", "15일") and not answers_agree("", "")
    assert answers_agree("가나", "가나")
    assert not answers_agree("가", "나")                                  # 1자 답은 포함 규칙을 쓰지 않는다
    assert answer_overlap("답 [abc123]", "답") == answer_overlap("답", "답")  # 인용 표기 무시
    assert answer_overlap("", "") == 0.0


# ---- oracle ----------------------------------------------------------------------------------------------------------------
def rec(cid, q, ans, gt, **kw):
    return QARecord(cid, q, ("d1",), answer=ans, evidence=gt, gt_contents=(gt,), **kw)


def test_oracle_filter_keeps_recoverable_and_rejects_others():
    calls = []
    def llm(prompt):
        calls.append(prompt)
        if "백업" in prompt: return "매일 새벽 2시에 전체 백업을 수행한다"
        if "주차" in prompt: return "알 수 없음"
        return "완전히 다른 내용의 답변입니다 하하하"
    drafts = [rec("a", "백업 시간은?", "매일 새벽 2시에 전체 백업을 수행한다", "매일 새벽 2시에 전체 백업을 수행한다."),
              rec("b", "주차 규정은?", "지정 구역만 가능", "주차는 지정 구역에서만 가능하다."),
              rec("c", "휴가는?", "연차 15일", "연차는 15일이다."),
              QARecord("u", "우주선?", (), should_fallback=True)]
    kept, rej, st = oracle_filter(llm, drafts)
    assert [r.case_id for r in kept] == ["a", "u"] and {r.case_id: r.reason for r in rej} == {"b": "oracle_unanswerable", "c": "oracle_mismatch"}
    assert all(r.status == "rejected" for r in rej) and st == {"passed": 1, "oracle_unanswerable": 1, "oracle_mismatch": 1, "skipped": 1}
    assert len(calls) == 3 and "우주선" not in "".join(calls)        # 답 없는 질문은 호출 대상이 아니다
    assert "매일 새벽 2시에" in calls[0] and "주차는 지정" not in calls[0]  # 근거 텍스트만 준다(다른 질문의 근거가 섞이지 않음)


def test_oracle_filter_llm_error_keeps_record_with_flag():
    def boom(p): raise RuntimeError("down")
    kept, rej, st = oracle_filter(boom, [rec("a", "q?", "x", "근거 텍스트 입니다")])
    assert [r.case_id for r in kept] == ["a"] and "oracle_error" in kept[0].flags and not rej and st == {"oracle_error": 1}


def test_oracle_threshold_is_respected():
    llm = lambda p: "매일 새벽 2시에 전체 백업을 수행합니다"                      # 포함 관계가 아니고 dice ≈ 0.76
    r = lambda: [rec("a", "q?", "매일 새벽 2시에 전체 백업을 수행한다", "근거")]
    assert oracle_filter(llm, r(), threshold=0.3)[0] and not oracle_filter(llm, r(), threshold=0.99)[0]


def test_generate_qa_records_qtype_and_old_jsonl_without_qtype_loads(tmp_path):
    from intellectagent.models import Chunk
    chunks = [Chunk("c" * 20, "d1", "연차휴가는 입사 후 1년이 지나면 15일이 부여된다. 병가는 연 60일까지 사용할 수 있다. " * 2, "length", 0, {})]
    def llm(p):
        if p.startswith("당신은 문서 검색"): return "연차휴가는 입사 후 며칠이 부여되나요?"
        return "15일이 부여된다"
    res = generate_qa(llm, chunks, 1)
    assert res.drafts and res.drafts[0].qtype == "numeric"
    write_jsonl(res.drafts, tmp_path / "q.jsonl")
    assert read_jsonl(tmp_path / "q.jsonl")[0].qtype == "numeric"
    old = json.loads((tmp_path / "q.jsonl").read_text(encoding="utf-8").splitlines()[0]); old.pop("qtype")
    (tmp_path / "old.jsonl").write_text(json.dumps(old, ensure_ascii=False) + "\n", encoding="utf-8")
    assert read_jsonl(tmp_path / "old.jsonl")[0].qtype == ""            # 옛 파일 호환


# ---- real_eval 순수 함수 --------------------------------------------------------------------------------------------------
def test_overlap_band_edges():
    assert [overlap_band(x) for x in (0.0, 0.49, 0.5, 0.79, 0.8, 1.0)] == ["low", "low", "mid", "mid", "high", "high"]


def test_stratify_groups_and_metrics():
    out = stratify([("a", 1), ("a", None), ("b", 2), ("b", 6)], 5)
    assert out["a"] == {"hit": 0.5, "mrr": 0.5, "n": 2} and out["b"]["hit"] == 0.5 and out["b"]["mrr"] == 0.25 and list(out) == ["a", "b"]


@pytest.mark.parametrize("kw,exp", [
    (dict(expected_fallback=True, fallback=True), "correct_refusal"),
    (dict(expected_fallback=True, fallback=False), "hallucinated_answer"),
    (dict(expected_fallback=False, fallback=True, gold_in_hits=False), "retrieval_miss"),
    (dict(expected_fallback=False, fallback=False, gold_in_hits=False), "retrieval_miss"),
    (dict(expected_fallback=False, fallback=True, gold_in_hits=True), "refused"),
    (dict(expected_fallback=False, fallback=False, gold_in_hits=True, gold_cited=False), "wrong_source"),
    (dict(expected_fallback=False, fallback=False, gold_in_hits=True, gold_cited=True, answer_overlap=0.1), "answer_mismatch"),
    (dict(expected_fallback=False, fallback=False, gold_in_hits=True, gold_cited=True, answer_overlap=0.3), "ok"),
    (dict(expected_fallback=False, fallback=False, gold_in_hits=True, gold_cited=True, answer_overlap=None), "ok"),
])
def test_classify_outcome(kw, exp):
    base = dict(gold_in_hits=False, gold_cited=False, answer_overlap=None); base.update(kw)
    assert classify_outcome(**base) == exp


def test_pareto_frontier():
    pts = [("fast_bad", 0.5, 10), ("slow_good", 0.9, 500), ("dominated", 0.4, 600), ("mid", 0.7, 100), ("tie_a", 0.7, 100)]
    assert pareto_frontier(pts) == {"fast_bad", "slow_good", "mid", "tie_a"}      # 동점 둘은 서로 지배하지 않는다
    assert pareto_frontier([]) == set()


def test_never_found():
    r = {"x": {"q1": 1, "q2": None, "q3": 9}, "y": {"q1": None, "q2": None, "q3": 2}}
    assert never_found(r, 5) == ["q2"]                                               # q3: y 가 2위로 찾았다 / q1: x 가 찾았다
    assert never_found({}, 5) == []


# ---- 스크립트 통합 -------------------------------------------------------------------------------------------------------
T1 = "서버 백업 지침: 매일 새벽 2시에 전체 백업을 수행하고 30일간 보관한다. 복구 훈련은 분기마다 실시한다. " * 4
T2 = "원격 근무 VPN 접속 절차: 사내 인증서를 설치한 뒤 이중 인증을 통과해야 접속할 수 있다. " * 4


def setup(tmp_path):
    root = tmp_path / "docs"; root.mkdir()
    (root / "a.md").write_text(T1, encoding="utf-8"); (root / "b.md").write_text(T2, encoding="utf-8")
    ids = {d.metadata["name"]: d.doc_id for d in map(SafeDocumentLoader(root, (".md",), 10**6).load, sorted(root.glob("*.md")))}
    qa = [QARecord("q1", "서버 백업은 언제 수행하나요", (ids["a.md"],), answer="매일 새벽 2시", qtype="factoid", lexical_overlap=0.9, status="approved"),
          QARecord("q2", "VPN 접속에 필요한 인증은 무엇인가요", (ids["b.md"],), answer="이중 인증", qtype="definition", lexical_overlap=0.6, status="approved"),
          QARecord("q3", "완전히 무관한 질문 입니다", (ids["a.md"],), qtype="factoid", lexical_overlap=0.1, status="approved"),   # 일부러 못 찾을 질문
          QARecord("u1", "우주선 엔진은?", (), should_fallback=True, qtype="unanswerable", status="approved")]
    write_jsonl(qa, tmp_path / "qa.jsonl")
    return root


def run(tmp_path, root, *extra):
    return ree.main(["--root", str(root), "--qa", str(tmp_path / "qa.jsonl"), "--out", str(tmp_path / "out"), "--embedder", "hash",
                     "--chunks", "100:10", "--retrievers", "bm25,hybrid_rrf", "--top-k", "2", "--bootstrap", "100", "--skip-ollama", *extra])


def test_script_reports_strata_pareto_and_never_found(tmp_path):
    run(tmp_path, setup(tmp_path))
    info = json.loads((tmp_path / "out/real_eval.json").read_text(encoding="utf-8"))
    best = next(c for c in info["candidates"] if c["candidate"] == info["best"])
    assert set(best["by_qtype"]) == {"factoid", "definition"} and set(best["by_overlap"]) == {"high", "mid", "low"}
    assert best["by_qtype"]["factoid"]["n"] == 2 and "_ranks" not in best
    assert info["pareto"] and set(info["pareto"]) <= {c["candidate"] for c in info["candidates"]}
    rep = (tmp_path / "out/report.md").read_text(encoding="utf-8")
    assert "질문 유형별" in rep and "어휘 겹침 층별" in rep and "Pareto" in rep
    assert isinstance(info["never_found"], list)


def test_e2e_outcomes_with_fake_generator(tmp_path):
    from intellectagent.chunk_grid import build_candidates
    from intellectagent.embeddings import HashEmbedder
    from intellectagent.qa_gen import load_approved
    root = setup(tmp_path)
    docs = [SafeDocumentLoader(root, (".md",), 10**6).load(p) for p in sorted(root.glob("*.md"))]
    cand = build_candidates(docs, ree.parse_chunks("100:10"), ree.parse_specs("hybrid_rrf", 0.5), embedder=HashEmbedder(), gate=True)[0]

    class Gen:
        def generate(self, q, hits): return f"매일 새벽 2시에 전체 백업을 수행한다 [{hits[0].chunk_id}]"
    recs = load_approved(tmp_path / "qa.jsonl")
    res = ree.run_e2e(cand, recs, "m", 2, -1.0, 4, tmp_path / "e2e", generator=Gen())     # 임계값 -1: gate 가 막지 않게
    assert sum(res["outcomes"].values()) == res["answered_or_fallback"] == 4
    assert res["outcomes"].get("hallucinated_answer") == 1                                # u1: 답 없는 질문에 답함
    rows = [json.loads(l) for l in (tmp_path / "e2e/e2e_samples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["case_id"]: r["qtype"] for r in rows}["q2"] == "definition" and all("outcome" in r for r in rows)
    assert "unanswerable" in res["outcomes_by_qtype"]


# ---- 변이 검사에서 살아남은 부분 ---------------------------------------------------------------------------------------------
def test_oracle_skips_unanswerable_even_if_it_carries_an_answer_and_uses_evidence_not_answer():
    calls = []
    def llm(p): calls.append(p); return "새벽 02:00 에 전체 백업"
    un = QARecord("u", "우주선 엔진?", (), should_fallback=True, answer="있음", gt_contents=("엉뚱한 근거",))
    r = QARecord("a", "백업 시간?", ("d1",), answer="매일 2시", evidence="새벽 02:00 에 전체 백업", gt_contents=("새벽 02:00 에 전체 백업",))
    kept, rej, st = oracle_filter(llm, [un, r])
    assert len(calls) == 1 and "엉뚱한 근거" not in calls[0]                 # 답 없는 질문은 답이 있어도 호출하지 않는다
    assert "새벽 02:00 에 전체 백업" in calls[0] and "매일 2시" not in calls[0]  # 근거 텍스트를 주고 기존 답은 주지 않는다(정답 유출 방지)


class _Fixed:
    def __init__(self, name, hits): self.name, self.hits = name, hits
    def search(self, q, k): return self.hits[:k]


def test_e2e_flags_wrong_source_when_citing_a_non_gold_document(tmp_path):
    gold = SearchHit("a" * 20, "GOLD", "금 본문", 0.8, {"gate_score": 0.8})
    other = SearchHit("b" * 20, "OTHER", "다른 본문", 0.9, {"gate_score": 0.9})
    rec = QARecord("q", "질문 입니다 정말", ("GOLD",), answer="금 본문", status="approved")
    class Gen:
        def generate(self, q, hits): return "금 본문 [%s]" % ("b" * 20)       # 정답 문서가 검색됐지만 다른 문서를 인용
    res = ree.run_e2e(_Fixed("x", [other, gold]), [rec], "m", 2, 0.0, 1, tmp_path / "e2e_ws", generator=Gen())
    assert res["outcomes"] == {"wrong_source": 1}
    class Gen2:
        def generate(self, q, hits): return "금 본문 [%s]" % ("a" * 20)
    assert ree.run_e2e(_Fixed("x", [other, gold]), [rec], "m", 2, 0.0, 1, tmp_path / "e2e_ws", generator=Gen2())["outcomes"] == {"ok": 1}


def test_pareto_uses_mrr_not_hit(tmp_path, monkeypatch):
    orig = ree.evaluate_candidate
    fake = iter([dict(hit=0.5, mrr=0.9, latency_ms_p50=10.0), dict(hit=0.9, mrr=0.5, latency_ms_p50=20.0)])
    def patched(*a, **k):
        r = orig(*a, **k); r.update(next(fake)); return r
    monkeypatch.setattr(ree, "evaluate_candidate", patched)
    run(tmp_path, setup(tmp_path))
    info = json.loads((tmp_path / "out/real_eval.json").read_text(encoding="utf-8"))
    first = info["candidates"][0]["candidate"]
    assert info["pareto"] == [first]       # MRR 기준이면 첫 후보가 둘째를 지배한다. hit 기준이면 둘 다 남는다


def test_e2e_records_why_it_refused_score_gate_vs_grounding(tmp_path):
    """모두 거절됐을 때 '점수 문턱' 때문인지 '인용 형식 검증' 때문인지 구분해 기록한다(원인 진단용)."""
    gold = SearchHit("a" * 20, "GOLD", "금 본문", 0.8, {"gate_score": 0.8})
    rec = QARecord("q", "질문 입니다 정말", ("GOLD",), answer="금 본문", status="approved")
    class NoCite:
        def generate(self, q, hits): return "금 본문입니다"                   # 인용 없음 → 검증 탈락
    r1 = ree.run_e2e(_Fixed("x", [gold]), [rec], "m", 2, 0.0, 1, tmp_path / "e1", generator=NoCite())
    assert r1["refusal_reasons"] == {"grounding_failed": 1}
    row = json.loads((tmp_path / "e1/e2e_samples.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["refusal_reason"] == "grounding_failed" and row["raw_answer"] == "금 본문입니다" and row["grounding"]["cited_ok"] == 0
    class Never:
        def generate(self, q, hits): raise AssertionError("점수 문턱에서 막혀야 생성기에 닿지 않는다")
    r2 = ree.run_e2e(_Fixed("x", [gold]), [rec], "m", 2, 0.99, 1, tmp_path / "e2", generator=Never())
    assert r2["refusal_reasons"] == {"score_gate": 1}

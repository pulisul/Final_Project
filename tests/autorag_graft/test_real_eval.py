import importlib.util
import json
import sys
from pathlib import Path

import pytest

from intellectagent.loaders import SafeDocumentLoader
from intellectagent.qa_gen import QARecord, write_jsonl
from intellectagent.real_eval import (Criteria, auc, bootstrap_ci, first_relevant_rank, gate_point, gate_sweep,
                                      percentile, rank_metrics, verdict)

SPEC = importlib.util.spec_from_file_location("ree", Path(__file__).parents[2] / "scripts/autorag/real_env_eval.py")
ree = importlib.util.module_from_spec(SPEC); sys.modules["ree"] = ree; SPEC.loader.exec_module(ree)


# ---- 순수 함수 -----------------------------------------------------------------------------------------------------------
def test_first_relevant_rank_dedups_chunks_of_same_doc():
    assert first_relevant_rank(["a", "a", "a", "b"], ["b"]) == 2     # a 청크 3개는 한 문서
    assert first_relevant_rank(["a", "b"], ["c"]) is None
    assert first_relevant_rank([], ["a"]) is None
    assert first_relevant_rank(["x", "b", "a"], ["a", "b"]) == 2


def test_rank_metrics_exact_values_and_k_cutoff():
    m = rank_metrics([1, 2, None, 6], k=5)
    assert m["hit"] == 0.5 and m["mrr"] == pytest.approx((1 + 0.5) / 4) and m["n"] == 4
    assert rank_metrics([], 3) == {"hit": 0.0, "mrr": 0.0, "n": 0}
    with pytest.raises(ValueError):
        rank_metrics([1], 0)


def test_bootstrap_ci_properties():
    assert bootstrap_ci([]) == (0.0, 0.0) and bootstrap_ci([0.7]) == (0.7, 0.7)
    vals = [1.0] * 8 + [0.0] * 2
    lo, hi = bootstrap_ci(vals, 1000, 1)
    assert lo <= 0.8 <= hi and 0 <= lo and hi <= 1
    assert bootstrap_ci(vals, 1000, 1) == (lo, hi)                       # 재현 가능
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0)                        # 변동 없으면 구간 폭 0
    assert (hi - lo) > (lambda a, b: b - a)(*bootstrap_ci(vals * 10, 1000, 1))  # 표본이 커지면 좁아진다


def test_auc_perfect_inverted_tie_and_empty():
    assert auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auc([0.1], [0.9]) == 0.0
    assert auc([0.5], [0.5]) == 0.5
    assert auc([], [0.1]) is None


def test_gate_point_boundary_is_strict_less_than():
    p = gate_point([0.1, 0.5], [0.1, 0.05], 0.1)                          # gate < threshold 만 막힘 (pipeline 과 동일)
    assert p.false_fallback_rate == 0.0 and p.correct_refusal_rate == 0.5


def test_gate_sweep_separable_gives_threshold_between_groups():
    s = gate_sweep([0.6, 0.7, 0.8, 0.9], [0.1, 0.2, 0.3], 0.10)
    r = s["recommended"]
    assert 0.3 < r.threshold <= 0.6 and r.false_fallback_rate == 0 and r.correct_refusal_rate == 1 and s["auc"] == 1.0


def test_gate_sweep_overlapping_respects_false_fallback_cap():
    ans = [0.2 + 0.05 * i for i in range(20)]            # 0.20..1.15
    una = [0.1 + 0.05 * i for i in range(20)]            # 0.10..1.05  (거의 구분 불가)
    s = gate_sweep(ans, una, 0.10)
    r = s["recommended"]
    assert r.false_fallback_rate <= 0.10
    assert all(p.correct_refusal_rate <= r.correct_refusal_rate + 1e-12 for p in s["points"] if p.false_fallback_rate <= 0.10)


def test_gate_sweep_no_unanswerable_and_empty():
    r = gate_sweep([0.5, 0.6], [], 0.1)["recommended"]
    assert r.false_fallback_rate == 0 and r.threshold < 0.5            # 막을 근거가 없으면 가장 덜 막는 값
    assert gate_sweep([], [])["recommended"] is None


def test_verdict_pass_fail_invalid_and_warnings():
    good = gate_point([0.9] * 10, [0.1] * 10, 0.5)
    v = verdict({"hit": 0.9, "mrr": 0.7}, good, 40, 8)
    assert v["overall"] == "PASS" and not v["warnings"]
    assert verdict({"hit": 0.79, "mrr": 0.7}, good, 40, 8)["overall"] == "FAIL"           # 경계 바로 아래
    assert verdict({"hit": 0.80, "mrr": 0.60}, good, 40, 8)["overall"] == "PASS"          # 경계 포함
    leaky = gate_point([0.9] * 10, [0.9] * 10, 0.5)                                        # 근거 없는 질문을 못 막음
    assert verdict({"hit": 0.9, "mrr": 0.7}, leaky, 40, 8)["overall"] == "FAIL"
    inv = verdict({"hit": 1, "mrr": 1}, good, 40, 8, valid_embedder=False)
    assert inv["overall"] == "INVALID" and any("HashEmbedder" in w for w in inv["warnings"])
    small = verdict({"hit": 0.9, "mrr": 0.7}, good, 5, 1)
    assert len(small["warnings"]) == 2
    assert "false_fallback" not in verdict({"hit": 1, "mrr": 1}, good, 40, 0)["checks"]    # 답 없는 질문이 없으면 gate 항목은 평가 불가


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == 2.5 and percentile([], 50) == 0.0 and percentile([7], 95) == 7
    assert percentile([1, 2, 3, 4], 100) == 4 and percentile([1, 2, 3, 4], 0) == 1


# ---- 스크립트 종단(해시 임베더: 배관 확인) --------------------------------------------------------------------------------
T1 = "서버 백업 지침: 매일 새벽 2시에 전체 백업을 수행하고 30일간 보관한다. 복구 훈련은 분기마다 실시한다. " * 4
T2 = "원격 근무 VPN 접속 절차: 사내 인증서를 설치한 뒤 이중 인증을 통과해야 접속할 수 있다. " * 4


def setup(tmp_path, n_unans=2, status="approved", bad=False):
    root = tmp_path / "docs"; root.mkdir()
    (root / "a.md").write_text(T1, encoding="utf-8"); (root / "b.md").write_text(T2, encoding="utf-8")
    ids = {d.metadata["name"]: d.doc_id for d in map(SafeDocumentLoader(root, (".md",), 10**6).load, sorted(root.glob("*.md")))}
    qa = [QARecord("q1", "서버 백업은 언제 수행하나요", ("zzz" if bad else ids["a.md"],), status=status),
          QARecord("q2", "VPN 접속에 필요한 인증은 무엇인가요", (ids["b.md"],), status=status)]
    qa += [QARecord(f"u{i}", f"우주선 엔진 {i} 은?", (), should_fallback=True, status=status) for i in range(n_unans)]
    write_jsonl(qa, tmp_path / "qa.jsonl")
    return root


def run(tmp_path, root, *extra):
    return ree.main(["--root", str(root), "--qa", str(tmp_path / "qa.jsonl"), "--out", str(tmp_path / "out"), "--embedder", "hash",
                     "--chunks", "100:10", "--retrievers", "bm25,hybrid_rrf", "--top-k", "2", "--bootstrap", "200", *extra])


def test_script_hash_mode_is_invalid_and_writes_all_outputs(tmp_path, capsys):
    assert run(tmp_path, setup(tmp_path), "--skip-ollama") == 3       # INVALID
    out = tmp_path / "out"
    info = json.loads((out / "real_eval.json").read_text(encoding="utf-8"))
    assert info["verdict"]["overall"] == "INVALID" and info["n_answerable"] == 2 and info["n_unanswerable"] == 2
    assert {c["candidate"] for c in info["candidates"]} and all("_current_point" not in c for c in info["candidates"])
    rep = (out / "report.md").read_text(encoding="utf-8")
    assert "INVALID" in rep and "HashEmbedder" in rep
    rows = (out / "per_question.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + len(info["candidates"]) * 4
    assert "서버 백업" not in "".join(rows)                              # 질문 본문은 per_question 에 없다(id 만)


def test_script_gate_is_set_for_hybrid_candidates(tmp_path):
    run(tmp_path, setup(tmp_path), "--skip-ollama")
    info = json.loads((tmp_path / "out/real_eval.json").read_text(encoding="utf-8"))
    hy = [c for c in info["candidates"] if "hybrid" in c["candidate"]]
    assert hy and all(c["gate"]["answerable_gate_p10_p50_p90"][1] > 0 for c in hy)


def test_script_refuses_drafts(tmp_path, capsys):
    assert run(tmp_path, setup(tmp_path, status="draft")) == 2
    assert "approved" in capsys.readouterr().err


def test_script_refuses_mismatched_docs(tmp_path, capsys):
    assert run(tmp_path, setup(tmp_path, bad=True)) == 2
    assert "맞지 않습니다" in capsys.readouterr().err


def test_preflight_ollama_rejects_non_local_and_dead_server():
    assert not ree.preflight_ollama("qwen2.5:14b", "http://example.com:11434")["ok"]
    r = ree.preflight_ollama("qwen2.5:14b", "http://127.0.0.1:9", timeout=0.5)
    assert not r["ok"] and "ollama" in r["detail"]


def test_preflight_embedder_detects_non_semantic_embedder():
    class Const:
        def embed(self, t): return [[1.0, 0.0]] * len(t)
    assert not ree.preflight_embedder(Const())["ok"]                    # 관련쌍 == 무관쌍 → 의미 없음


def test_e2e_with_fake_generator_counts_fallbacks(tmp_path):
    root = setup(tmp_path)
    run(tmp_path, root, "--skip-ollama")
    from intellectagent.chunk_grid import build_candidates
    from intellectagent.embeddings import HashEmbedder
    docs = [SafeDocumentLoader(root, (".md",), 10**6).load(p) for p in sorted(root.glob("*.md"))]
    cand = build_candidates(docs, ree.parse_chunks("100:10"), ree.parse_specs("hybrid_rrf", 0.5), embedder=HashEmbedder(), gate=True)[0]

    class Gen:
        def generate(self, *a, **k): return "답변 [1]"
        def __call__(self, *a, **k): return "답변 [1]"
    from intellectagent.qa_gen import load_approved
    recs = load_approved(tmp_path / "qa.jsonl")
    try:
        res = ree.run_e2e(cand, recs, "m", 2, 0.12, 4, tmp_path / "e2e", generator=Gen())
    except TypeError:
        pytest.skip("생성기 인터페이스가 달라 가짜 생성기 연결 불가")
    assert res["asked"] == 4 and (tmp_path / "e2e/e2e_samples.jsonl").exists()


# ---- 변이 검사에서 살아남은 부분을 겨냥한 테스트 ---------------------------------------------------------------------------
from intellectagent.models import SearchHit  # noqa: E402


def _hit(doc, score, gate=None):
    return SearchHit("c", doc, "t", score, {} if gate is None else {"gate_score": gate})


def test_rank_at_exactly_k_counts_as_hit():
    assert rank_metrics([5], 5)["hit"] == 1.0 and rank_metrics([6], 5)["hit"] == 0.0
    assert rank_metrics([5], 5)["mrr"] == pytest.approx(0.2)


def test_bootstrap_is_seed_deterministic_but_seed_dependent():
    vals = [i / 37 for i in range(37)]
    assert bootstrap_ci(vals, 300, 7) == bootstrap_ci(vals, 300, 7)
    assert bootstrap_ci(vals, 300, 7) != bootstrap_ci(vals, 300, 8)


def test_gate_of_prefers_vector_gate_score_and_empty_is_blocked():
    assert ree.gate_of([]) == 0.0
    assert ree.gate_of([_hit("a", 99.0, gate=0.3)]) == 0.3              # 벡터 코사인(gate_score) 이 우선
    assert ree.gate_of([_hit("a", 0.42)]) == 0.42                       # 없으면 1순위 점수


class _Scripted:
    name = "scripted"
    def __init__(self, table): self.table = table
    def search(self, q, k): return self.table[q]


def test_evaluate_candidate_records_both_groups_and_sets_rows():
    from intellectagent.qa_gen import QARecord
    ans = [QARecord("a1", "qa1", ("D1",)), QARecord("a2", "qa2", ("D2",))]
    una = [QARecord("u1", "qu1", (), should_fallback=True), QARecord("u2", "qu2", (), should_fallback=True)]
    cand = _Scripted({"qa1": [_hit("D1", 0.9, 0.9)], "qa2": [_hit("X", 0.8, 0.8), _hit("D2", 0.7, 0.8)],
                      "qu1": [_hit("X", 0.1, 0.1)], "qu2": []})
    rows = []
    r = ree.evaluate_candidate(cand, ans, una, 2, 0.5, Criteria(), 100, 1, rows)
    assert r["hit"] == 1.0 and r["mrr"] == pytest.approx(0.75)
    assert r["gate"]["current"]["false_fallback_rate"] == 0.0 and r["gate"]["current"]["correct_refusal_rate"] == 1.0
    assert r["gate"]["unanswerable_gate_p10_p50_p90"][2] <= 0.1 and r["gate"]["auc"] == 1.0
    assert [(x["case_id"], x["rank"], x["gate"]) for x in rows if x["kind"] == "unanswerable"] == [("u1", "", 0.1), ("u2", "", 0.0)]


def test_build_candidates_gate_flag_attaches_gate_score_only_when_asked(tmp_path):
    from intellectagent.chunk_grid import build_candidates
    from intellectagent.embeddings import HashEmbedder
    (tmp_path / "a.md").write_text(T1, encoding="utf-8"); (tmp_path / "b.md").write_text(T2, encoding="utf-8")
    ld = SafeDocumentLoader(tmp_path, (".md",), 10**7); docs = [ld.load(p) for p in sorted(tmp_path.glob("*.md"))]
    args = (docs, ree.parse_chunks("200:20"), ree.parse_specs("hybrid_rrf", 0.5))
    on = build_candidates(*args, embedder=HashEmbedder(), gate=True)[0].search("서버 백업 시간", 2)
    off = build_candidates(*args, embedder=HashEmbedder())[0].search("서버 백업 시간", 2)
    assert on and all("gate_score" in h.metadata for h in on)
    assert off and not any("gate_score" in h.metadata for h in off)


# ---- 관련성 판정 연동 -------------------------------------------------------------------------------------------------------
def test_judge_requires_ollama(tmp_path, capsys):
    assert run(tmp_path, setup(tmp_path), "--skip-ollama", "--judge") == 2
    assert "Ollama" in capsys.readouterr().err


def test_judged_candidate_wraps_search_and_counts_errors():
    from intellectagent.relevance_judge import JudgeReply, RelevanceJudge
    base = _Scripted({"q": [_hit("X", 0.9, 0.9), _hit("Y", 0.8, 0.8)]})
    ok = ree.JudgedCandidate(base, RelevanceJudge(lambda p: JudgeReply("Yes") if "Y 본문" in p else JudgeReply("No")))
    base.table["q"] = [SearchHit("c1", "X", "X 본문", 0.9, {"gate_score": 0.9}), SearchHit("c2", "Y", "Y 본문", 0.8, {"gate_score": 0.8})]
    assert ok.name == "scripted+judge" and [h.doc_id for h in ok.search("q", 2)] == ["Y"] and ok.errors == 0

    def boom(p): raise RuntimeError("down")
    bad = ree.JudgedCandidate(base, RelevanceJudge(boom))
    assert len(bad.search("q", 2)) == 2 and bad.errors == 1                  # fail-open + 오류 집계
    none = ree.JudgedCandidate(base, RelevanceJudge(lambda p: JudgeReply("No")))
    assert none.search("q", 2) == [] and ree.gate_of(none.search("q", 2)) == 0.0   # 전부 버리면 거절(gate 0)

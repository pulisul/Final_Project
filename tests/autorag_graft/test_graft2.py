"""2차 이식분 테스트: 토큰 지표, hybrid_cc, 청킹x검색 격자, QA 자동 생성, HyDE, 파싱 캐시."""
import json
import re
from pathlib import Path

import pytest

from intellectagent import qa_gen as Q
from intellectagent import retrieval_contents as RC
from intellectagent.chunk_grid import RetrieverSpec, build_candidates, chunk_label
from intellectagent.chunk_strategies import ChunkStrategyConfig
from intellectagent.grid_search import Candidate, run_grid
from intellectagent.hashing import sha256_text
from intellectagent.hybrid import HybridRetriever, cc_fuse
from intellectagent.lexical import BM25Index, tok_char_bigram
from intellectagent.loaders import SafeDocumentLoader, UnsafePathError, UnsupportedFileError
from intellectagent.models import Chunk, Document, EvalCase, SearchHit
from intellectagent.parse_cache import CachingDocumentLoader, ParseCache
from intellectagent.query_expansion import HydeRetriever, hyde_passage


def H(cid, score, doc=None, text=None):
    return SearchHit(cid, doc or cid, text or f"text-{cid}", score)


def GEN(*a, **kw):
    kw.setdefault("min_chars", 20)  # 테스트 문서는 짧다(운영 기본값은 80자)
    return Q.generate_qa(*a, **kw)


def mk_chunk(cid, doc, text, ordinal=0):
    return Chunk(cid, doc, text, "length", ordinal)


# ---- 토큰 지표 --------------------------------------------------------------------------------------------------------
def test_token_metrics_hand_computed():
    gt = ["서버 백업 매일 실행"]
    pred = ["서버 백업 실행 로그 보관", "전혀 다른 문장"]
    p, r = 3 / 5, 3 / 4  # 첫 청크: 공통 3, pred 5, gt 4 / 둘째 청크: 0
    f = 2 * p * r / (p + r)
    assert RC.retrieval_token_precision(gt, pred) == pytest.approx(p / 2)
    assert RC.retrieval_token_recall(gt, pred) == pytest.approx(r / 2)
    assert RC.retrieval_token_f1(gt, pred) == pytest.approx(f / 2)


def test_token_metrics_normalization_and_empty():
    assert RC.normalize_string("The Server, a BACKUP!") == "server backup"
    assert RC.retrieval_token_f1(["x"], []) == 0.0
    assert RC.retrieval_token_f1([], ["x"]) == 0.0
    assert RC.retrieval_token_recall([["서버 백업"]], ["서버 백업 서버"]) == pytest.approx(1.0)  # 중첩 정답 리스트도 평탄화


def test_token_metrics_custom_tokenizer_absorbs_korean_particles():
    gt, pred = ["서버는 백업된다"], ["서버를 백업한다"]
    assert RC.retrieval_token_recall(gt, pred) == 0.0                                  # 어절 단위: 조사가 달라 0
    assert RC.retrieval_token_recall(gt, pred, tokenizer=tok_char_bigram) > 0.0        # bigram: 공통 '서버','백업' 인식


def test_content_metric_requires_gt_contents_in_grid():
    cand = Candidate("c", lambda q, k: [H("a", 1.0)])
    with pytest.raises(ValueError):
        run_grid([cand], [EvalCase("1", "q", ("a",))], metrics=("retrieval_token_recall",))


def test_grid_uses_content_metrics_when_gt_contents_present():
    c = Q.QARecord("1", "q", ("a",), gt_contents=("서버 백업",))
    cand = Candidate("c", lambda q, k: [H("a", 1.0, text="서버 백업 로그")])
    row = run_grid([cand], [c], metrics=("retrieval_recall", "retrieval_token_recall", "retrieval_token_precision"))["rows"][0]
    assert row["retrieval_token_recall"] == pytest.approx(1.0)
    assert row["retrieval_token_precision"] == pytest.approx(2 / 3)
    assert row["retrieval_recall"] == pytest.approx(1.0)


# ---- hybrid_cc --------------------------------------------------------------------------------------------------------
def test_cc_fuse_hand_computed_mm():
    sem = [H("a", 0.9), H("b", 0.5), H("c", 0.1)]   # mm: 1, .5, 0
    lex = [H("b", 8.0), H("d", 4.0)]                 # mm: 1, 0
    out = cc_fuse(sem, lex, top_k=4, weight=0.7)
    got = {h.chunk_id: h.score for h in out}
    assert got["a"] == pytest.approx(0.7)
    assert got["b"] == pytest.approx(0.7 * 0.5 + 0.3 * 1.0)  # 0.65
    assert got["c"] == pytest.approx(0.0) and got["d"] == pytest.approx(0.0)
    assert [h.chunk_id for h in out][:2] == ["a", "b"]
    assert [h.chunk_id for h in out][2:] == ["c", "d"]  # 동점은 먼저 나온 순서(semantic 목록 우선)


def test_cc_weight_extremes_and_validation():
    sem, lex = [H("a", .9), H("b", .1)], [H("b", 5.0), H("c", 1.0)]
    assert cc_fuse(sem, lex, 3, weight=1.0)[0].chunk_id == "a"
    assert cc_fuse(sem, lex, 3, weight=0.0)[0].chunk_id == "b"
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            cc_fuse(sem, lex, 3, weight=bad)
    with pytest.raises(ValueError):
        cc_fuse(sem, lex, 3, weight=.5, normalize_method="nope")
    with pytest.raises(ValueError):
        cc_fuse(sem, lex, 0, weight=.5)


def test_cc_degenerate_scores_become_zero_like_original_nan_fill():
    sem, lex = [H("a", .5), H("b", .5)], [H("b", 3.0), H("c", 3.0)]
    for m in ("mm", "z", "dbsf"):
        assert all(h.score == 0.0 for h in cc_fuse(sem, lex, 3, weight=.5, normalize_method=m))


def test_cc_tmm_uses_theoretical_min():
    sem, lex = [H("a", 1.0), H("b", 0.0)], [H("b", 4.0)]
    # semantic tmm (min=-1): a=(1+1)/(1+1)=1, b=(0+1)/2=.5 ; lexical tmm (min=0): b=1
    got = {h.chunk_id: h.score for h in cc_fuse(sem, lex, 2, weight=.5, normalize_method="tmm")}
    assert got["a"] == pytest.approx(.5) and got["b"] == pytest.approx(.5 * .5 + .5 * 1.0)


def test_hybrid_retriever_cc_mode_and_degradation():
    class V:  # 벡터 쪽
        def search(self, q, k): return [H("a", .9), H("b", .2)]
    class L:
        def search(self, q, k): return [H("b", 7.0), H("c", 1.0)]
    class Boom:
        def search(self, q, k): raise RuntimeError("x")
    cc = HybridRetriever(V(), L(), fusion="cc", cc_weight=0.5)
    assert [h.chunk_id for h in cc.search("q", 3)] == ["a", "b", "c"]  # a=.5, b=.5(동점은 먼저 나온 a), c=0
    assert [h.chunk_id for h in HybridRetriever(Boom(), L(), fusion="cc").search("q", 2)] == ["b", "c"]
    assert HybridRetriever(Boom(), Boom()).search("q", 2) == []
    with pytest.raises(ValueError):
        HybridRetriever(V(), L(), fusion="nope")


# ---- 청킹 x 검색 격자 -------------------------------------------------------------------------------------------------
SENTS = [
    "서버 백업은 매일 새벽 두 시에 자동으로 실행됩니다.", "휴가 신청은 팀장 승인을 거쳐 인사시스템에서 진행합니다.",
    "출장비 정산은 영수증을 제출한 뒤 칠 일 이내에 처리됩니다.", "로그인 실패가 다섯 번 쌓이면 계정이 잠깁니다.",
    "회의실 예약은 사내 캘린더에서 하며 미사용 시 자동 취소됩니다.", "재택근무는 주 이 회까지 전날 신청하면 가능합니다.",
]


def make_docs():
    text = " ".join(SENTS)
    return [Document("docA", Path("a.md"), text, sha256_text(text), {"name": "a.md"})]


def test_chunk_grid_builds_all_combinations_and_compares_chunk_sizes():
    docs = make_docs()
    cfgs = [ChunkStrategyConfig("length", 400, 0), ChunkStrategyConfig("length", 60, 0)]
    specs = [RetrieverSpec("bm25"), RetrieverSpec("vector"), RetrieverSpec("hybrid_rrf"),
             RetrieverSpec("hybrid_cc", cc_weight=0.4, cc_normalize="mm")]
    cands = build_candidates(docs, cfgs, specs)
    assert len(cands) == len(cfgs) * len(specs)
    assert len({c.name for c in cands}) == len(cands)
    cases = [Q.QARecord("1", "로그인 실패가 쌓이면 계정은 어떻게 되나요", ("docA",), gt_contents=(SENTS[3],)),
             Q.QARecord("2", "서버 백업은 언제 실행되나요", ("docA",), gt_contents=(SENTS[0],))]
    res = run_grid(cands, cases, top_k=1, metrics=("retrieval_recall", "retrieval_token_recall", "retrieval_token_precision"))
    rows = {r["candidate"]: r for r in res["rows"]}
    big, small = chunk_label(cfgs[0]), chunk_label(cfgs[1])
    # 큰 청크는 근거를 통째로 담아 recall 이 높고, 작은 청크는 근거만 담아 precision 이 높다 — 이 지표가 크기 차이를 드러내야 한다.
    assert rows[f"{big}|bm25+char_bigram"]["retrieval_token_recall"] >= rows[f"{small}|bm25+char_bigram"]["retrieval_token_recall"]
    assert rows[f"{small}|bm25+char_bigram"]["retrieval_token_precision"] > rows[f"{big}|bm25+char_bigram"]["retrieval_token_precision"]
    assert all(r["param_chunk"] in (big, small) for r in res["rows"])
    assert rows[f"{big}|bm25+char_bigram"]["param_chunk_count"] < rows[f"{small}|bm25+char_bigram"]["param_chunk_count"]


def test_chunk_grid_validation_and_reranker_failure_is_not_fatal():
    docs = make_docs()
    with pytest.raises(ValueError):
        RetrieverSpec("nope")
    with pytest.raises(ValueError):
        RetrieverSpec("bm25", tokenizer="nope")
    with pytest.raises(KeyError):
        build_candidates(docs, [ChunkStrategyConfig("length", 100, 0)], [RetrieverSpec("bm25", reranker="r")])
    with pytest.raises(ValueError):
        build_candidates(docs, [ChunkStrategyConfig("length", 100, 0)] * 2, [RetrieverSpec("bm25")])

    class Bad:
        def rerank(self, q, hits): raise RuntimeError("model missing")
    class Reverse:
        def rerank(self, q, hits): return list(reversed(hits))
    cfg = [ChunkStrategyConfig("length", 60, 0)]
    q = "합니다 됩니다 가능합니다"  # 여러 청크에 걸리는 질의(BM25 는 점수 0 청크를 돌려주지 않는다)
    plain = build_candidates(docs, cfg, [RetrieverSpec("bm25")])[0].search(q, 3)
    assert len(plain) >= 2
    bad = build_candidates(docs, cfg, [RetrieverSpec("bm25", reranker="r")], rerankers={"r": Bad()})[0].search(q, 3)
    assert [h.chunk_id for h in bad] == [h.chunk_id for h in plain]  # 실패하면 리랭크 전 순서 유지
    full = build_candidates(docs, cfg, [RetrieverSpec("bm25", reranker="r", candidate_k=20)], rerankers={"r": Reverse()})[0]
    rev = full.search(q, 2)
    pool = build_candidates(docs, cfg, [RetrieverSpec("bm25")])[0].search(q, 20)
    assert [h.chunk_id for h in rev] == [h.chunk_id for h in reversed(pool)][:2]  # 후보 풀(candidate_k)을 리랭크한 뒤 top_k 로 자른다


# ---- QA 생성 ---------------------------------------------------------------------------------------------------------
DOC_TEXT = {"d1": "연차 휴가는 입사 1년 후부터 15일이 부여됩니다. 신청은 팀장 승인을 거쳐 인사시스템에서 합니다. 미사용 휴가는 소멸합니다.",
            "d2": "서버 백업은 매일 새벽 2시에 자동 실행됩니다. 사본은 다른 지역 저장소에 30일 동안 보관됩니다. 복구 훈련은 분기마다 합니다.",
            "d3": "출장비는 교통비와 숙박비로 나뉩니다. 영수증은 출장 후 7일 이내에 정산 담당자에게 제출해야 합니다. 미제출 시 지급이 보류됩니다."}
CHUNKS = [mk_chunk(f"c-{d}", d, t) for d, t in DOC_TEXT.items()]
QUESTION_FOR = {"d1": "신입 직원의 연차는 며칠 생기나요?", "d2": "백업 사본은 얼마 동안 두나요?", "d3": "출장 영수증은 언제까지 내야 하나요?"}
ANSWER_FOR = {"d1": "입사 1년 후부터 15일이 부여됩니다.", "d2": "30일 동안 보관됩니다.", "d3": "출장 후 7일 이내에 제출해야 합니다."}


def stub_llm(prompt: str) -> str:
    doc = next((d for d, t in DOC_TEXT.items() if t in prompt), None)
    if prompt.startswith("당신은 문서 검색"):
        return "질문: " + QUESTION_FOR[doc]
    if prompt.startswith("아래 TEXT만 근거로"):
        return ANSWER_FOR[doc]
    raise AssertionError("unexpected prompt")


def test_sample_chunks_deterministic_and_spread():
    many = [mk_chunk(f"{d}-{i}", d, "가" * 100, i) for d in ("a", "b", "c") for i in range(5)]
    s1, s2 = Q.sample_chunks(many, 6, seed=1), Q.sample_chunks(many, 6, seed=1)
    assert [c.chunk_id for c in s1] == [c.chunk_id for c in s2]
    assert sorted(c.doc_id for c in s1) == ["a", "a", "b", "b", "c", "c"]  # 문서 편중 없음
    assert [c.chunk_id for c in Q.sample_chunks(many, 6, seed=2)] != [c.chunk_id for c in s1]
    assert Q.sample_chunks([mk_chunk("x", "a", "짧다")], 3, min_chars=80) == []


def test_generate_qa_end_to_end_with_stub():
    res = GEN(stub_llm, CHUNKS, 3, seed=3)
    assert res.stats["drafts"] == 3 and not res.rejected
    by_doc = {r.relevant_doc_ids[0]: r for r in res.drafts}
    assert set(by_doc) == {"d1", "d2", "d3"}
    for d, r in by_doc.items():
        assert r.question == QUESTION_FOR[d] and r.status == "draft" and not r.should_fallback
        assert r.evidence and r.evidence in DOC_TEXT[d]               # 근거는 항상 청크의 부분 문자열
        assert r.gt_contents == (r.evidence,) and r.gold_chunk_id == f"c-{d}"
        assert 0.0 <= r.lexical_overlap <= 1.0
    assert by_doc["d2"].evidence.startswith("사본은")                   # '30일 동안 보관' 답과 가장 겹치는 문장
    ec = by_doc["d1"].to_eval_case()
    assert isinstance(ec, EvalCase) and ec.relevant_doc_ids == ("d1",)


def test_generate_qa_filters_and_dedupe():
    def llm(prompt):
        doc = next((d for d, t in DOC_TEXT.items() if t in prompt), None)
        if prompt.startswith("당신은 문서 검색"):
            return {"d1": "위 문서에서 연차는 며칠인가요?", "d2": "백업 사본은 얼마 동안 두나요?", "d3": "백업 사본은 얼마 동안 두나요?"}[doc]
        return {"d2": "알 수 없음", "d3": "30일입니다."}.get(doc, "x")
    res = GEN(llm, CHUNKS, 3, seed=3, attempts_factor=1)
    reasons = sorted(r.reason for r in res.rejected)
    assert reasons == ["dont_know", "passage_dependent"] and len(res.drafts) == 1
    assert res.drafts[0].relevant_doc_ids == ("d3",)
    # 중복 제거: 같은 질문이 두 번 나오면 두 번째는 duplicate
    res2 = GEN(lambda p: "백업 사본은 얼마 동안 두나요?" if p.startswith("당신은") else "30일입니다.", CHUNKS, 3, attempts_factor=1)
    assert [r.reason for r in res2.rejected] == ["duplicate", "duplicate"] and len(res2.drafts) == 1


def test_generate_qa_survives_llm_errors_and_dependency_llm():
    calls = {"n": 0}
    def flaky(p):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("ollama down")
        return stub_llm(p)
    res = GEN(flaky, CHUNKS, 2, attempts_factor=2)
    assert res.stats["llm_error"] == 1 and len(res.drafts) == 2
    judged = GEN(stub_llm, CHUNKS, 3, dependency_llm=lambda p: "True")
    assert not judged.drafts and {r.reason for r in judged.rejected} == {"passage_dependent_llm"}
    errs = GEN(stub_llm, CHUNKS, 3, dependency_llm=lambda p: (_ for _ in ()).throw(RuntimeError("x")))
    assert len(errs.drafts) == 3 and errs.stats["dependency_llm_error"] == 3


def test_rules_and_parsers():
    assert Q.is_passage_dependent("위 문서에서 휴가는 며칠인가요?") and Q.is_passage_dependent("본문에 따르면 얼마인가요?")
    assert not Q.is_passage_dependent("연차 휴가는 며칠 부여되나요?") and not Q.is_passage_dependent("문서 반출 시 암호화가 필요한가요?")
    assert Q.is_dont_know("모르겠습니다") and Q.is_dont_know("본문에서 확인할 수 없습니다") and not Q.is_dont_know("15일입니다")
    assert Q.parse_question('1. "휴가는 며칠인가요?"') == "휴가는 며칠인가요?"
    assert Q.parse_question("<think>음...</think>\n질문: 백업은 몇 시에 하나요?\n설명입니다") == "백업은 몇 시에 하나요?"
    assert Q.parse_question("") is None and Q.parse_question("네") is None and Q.parse_question("가 " * 120) is None
    assert Q.parse_bool("True.") is True and Q.parse_bool("false") is False and Q.parse_bool("글쎄") is None
    assert Q.lexical_overlap("서버 백업", "서버 백업은 매일") == 1.0 and Q.lexical_overlap("휴가", "서버") == 0.0


def test_evidence_is_empty_when_nothing_overlaps():
    assert Q.extract_evidence("서버 백업은 매일 실행됩니다. 휴가는 소멸합니다.", "zzz qqq", "xxx") == ""
    assert Q.extract_evidence("짧다", "짧다") == ""  # 8자 미만 문장은 근거 후보에서 제외


def test_no_evidence_falls_back_to_whole_chunk_with_flag():
    llm = lambda p: "특이한질문 내용이 무엇인가요?" if p.startswith("당신은") else "zzzz"
    r = GEN(llm, [CHUNKS[0]], 1).drafts[0]
    assert "no_evidence_sentence" in r.flags and r.gt_contents == (CHUNKS[0].text,)


def test_jsonl_roundtrip_and_only_approved_are_loaded(tmp_path: Path):
    res = GEN(stub_llm, CHUNKS, 3)
    p = tmp_path / "qa.jsonl"
    Q.write_jsonl(res.drafts, p)
    back = Q.read_jsonl(p)
    assert back == res.drafts
    assert Q.load_approved(p) == []                       # 검수 전 초안은 평가에 쓰이지 않는다
    lines = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    lines[0]["status"] = "approved"
    p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines), encoding="utf-8")
    ok = Q.load_approved(p)
    assert len(ok) == 1 and ok[0].case_id == lines[0]["case_id"]
    # 기존 load_cases 와도 호환(추가 필드는 무시)
    from intellectagent.evaluation import load_cases
    assert len(load_cases(p)) == 3


def test_unanswerable_generation_filters_by_corpus_coverage():
    bm25 = BM25Index(CHUNKS, tok_char_bigram, idf_variant="lucene")
    llm = lambda p: "서버 백업은 매일 하나요?\n우주 정거장 건설 예산은 얼마인가요?\n번호 없는 잡담"
    res = Q.generate_unanswerable(llm, ["a.md", "b.md"], 5, bm25, max_coverage=0.6)
    assert [r.question for r in res.drafts] == ["우주 정거장 건설 예산은 얼마인가요?"]
    assert res.drafts[0].should_fallback and "unanswerable_unverified" in res.drafts[0].flags and res.drafts[0].relevant_doc_ids == ()
    assert res.rejected[0].reason == "corpus_coverage_high"


def test_ollama_adapter_handles_dict_object_and_think_tags():
    class C1:
        def chat(self, model, messages, options): C1.seen = (model, messages[0]["content"], options); return {"message": {"content": "답"}}
    assert Q.ollama_llm("m", client=C1(), seed=7)("프롬프트") == "답" and C1.seen == ("m", "프롬프트", {"temperature": 0.0, "seed": 7})
    class Msg: content = "객체형"
    class R: message = Msg()
    class C2:
        def chat(self, **kw): return R()
    assert Q.ollama_llm("m", client=C2())("p") == "객체형"
    llm = Q.ollama_llm("m", client=type("C", (), {"chat": lambda self, **kw: {"message": {"content": "<think>생각</think>\n질문: 휴가는 며칠인가요?"}}})())
    assert Q.parse_question(llm("x")) == "휴가는 며칠인가요?"


# ---- HyDE ------------------------------------------------------------------------------------------------------------
def test_hyde_uses_passage_then_falls_back_and_can_fuse():
    seen = []
    class Inner:
        def search(self, q, k, *a, **kw):
            seen.append(q)
            return [H("p", 1.0)] if q == "가상 문단" else [H("o", 1.0)]
    assert [h.chunk_id for h in HydeRetriever(Inner(), lambda p: "가상 문단").search("질문", 2)] == ["p"]
    assert seen == ["가상 문단"]
    assert [h.chunk_id for h in HydeRetriever(Inner(), lambda p: "  ").search("질문", 2)] == ["o"]       # 빈 응답 -> 원 질문
    boom = lambda p: (_ for _ in ()).throw(RuntimeError("x"))
    assert [h.chunk_id for h in HydeRetriever(Inner(), boom).search("질문", 2)] == ["o"]                 # LLM 실패 -> 원 질문
    fused = HydeRetriever(Inner(), lambda p: "가상 문단", include_original=True).search("질문", 2)
    assert {h.chunk_id for h in fused} == {"p", "o"}
    assert hyde_passage(lambda p: p, "Q").endswith("Question: Q\nPassage:")  # 원본 프롬프트 형식


# ---- 파싱 캐시 -------------------------------------------------------------------------------------------------------
def make_loader(tmp_path: Path, **kw):
    root = tmp_path / "docs"; root.mkdir(exist_ok=True)
    cache = ParseCache(tmp_path / "cache")
    return root, cache, CachingDocumentLoader(SafeDocumentLoader(root, (".md", ".txt", ".docx"), 1_000_000), cache)


def test_parse_cache_hit_equals_uncached_document(tmp_path: Path):
    root, cache, loader = make_loader(tmp_path)
    f = root / "a.md"; f.write_text("# 제목\n\n서버 백업은 매일 합니다.\n", encoding="utf-8")
    plain = SafeDocumentLoader(root, (".md",), 1_000_000).load(f)
    first, second = loader.load(f), loader.load(f)
    assert first == plain and second == plain
    assert (cache.stats.misses, cache.stats.hits, cache.stats.writes) == (1, 1, 1)
    mirror = cache.mirror_path(plain.sha256, ".md")
    assert mirror.read_text(encoding="utf-8") == plain.text        # 사람이 열어 볼 수 있는 .md 가 정확히 파싱 결과


def test_parse_cache_detects_tampering_and_recovers(tmp_path: Path):
    root, cache, loader = make_loader(tmp_path)
    f = root / "a.md"; f.write_text("원본 내용입니다 서버 백업", encoding="utf-8")
    doc = loader.load(f)
    mirror = cache.mirror_path(doc.sha256, ".md")
    mirror.write_text("누군가 고친 내용", encoding="utf-8")
    assert cache.verify_all() == [str(mirror)]
    again = loader.load(f)
    assert again.text == "원본 내용입니다 서버 백업" and cache.stats.corrupt == 1   # 변조는 버리고 다시 파싱
    assert cache.verify_all() == [] and mirror.read_text(encoding="utf-8") == again.text


def test_parse_cache_content_change_and_fingerprint(tmp_path: Path):
    root, cache, loader = make_loader(tmp_path)
    f = root / "a.md"; f.write_text("v1 내용", encoding="utf-8")
    d1 = loader.load(f); f.write_text("v2 내용", encoding="utf-8"); d2 = loader.load(f)
    assert d1.sha256 != d2.sha256 and d2.text == "v2 내용" and cache.stats.hits == 0
    other = ParseCache(tmp_path / "cache", fingerprint="a" * 12)
    assert other.get(d2.sha256, ".md") is None                      # 파서 지문이 다르면 같은 파일이라도 미스
    other.put(d2.sha256, ".md", "x")
    assert cache.purge_other_versions() == 1 and not (tmp_path / "cache" / ("a" * 12)).exists()


def test_parse_cache_failures_not_cached_and_safety_checks_still_run(tmp_path: Path):
    root, cache, loader = make_loader(tmp_path)
    bad = root / "broken.docx"; bad.write_bytes(b"not a zip")
    with pytest.raises(UnsupportedFileError):
        loader.load(bad)
    assert cache.stats.writes == 0
    f = root / "a.md"; f.write_text("정상 문서 내용", encoding="utf-8"); loader.load(f)
    link = root / "link.md"; link.symlink_to(f)
    with pytest.raises(UnsafePathError):
        loader.load(link)                                            # 캐시에 있는 내용이라도 심볼릭 링크는 거부
    outside = tmp_path / "out.md"; outside.write_text("바깥", encoding="utf-8")
    with pytest.raises(UnsafePathError):
        loader.load(outside)


def test_parse_cache_inside_root_is_not_reingested(tmp_path: Path):
    root = tmp_path / "docs"; root.mkdir()
    cache = ParseCache(root / ".cache")                              # 일부러 허용 루트 안에 둔다
    loader = CachingDocumentLoader(SafeDocumentLoader(root, (".md",), 1_000_000), cache)
    f = root / "a.md"; f.write_text("정상 문서 내용입니다", encoding="utf-8"); loader.load(f)
    assert any(p.suffix == ".md" for p in (root / ".cache").rglob("*"))   # 미러가 실제로 생겼고
    assert loader.discover() == [f.resolve()] or loader.discover() == [f]  # 색인 대상에는 들어가지 않는다


def test_parse_cache_office_and_lossless_surrogates(tmp_path: Path):
    docx = pytest.importorskip("docx")
    root, cache, loader = make_loader(tmp_path)
    d = docx.Document(); d.add_paragraph("한글 문단 서버 백업"); t = d.add_table(rows=1, cols=2)
    t.cell(0, 0).text = "항목"; t.cell(0, 1).text = "값"
    path = root / "x.docx"; d.save(str(path))
    first, second = loader.load(path), loader.load(path)
    assert first == second and "서버 백업" in first.text and cache.stats.hits == 1
    s = "깨진\ud800문자"
    cache.put("b" * 64, ".md", s)
    assert cache.get("b" * 64, ".md") == s
    with pytest.raises(ValueError):
        cache.mirror_path("../../etc/passwd" + "0" * 48, ".md")       # 경로 조작 문자열은 해시로 인정하지 않는다
    with pytest.raises(ValueError):
        cache.mirror_path("c" * 64, ".md/../x")


def test_parse_cache_fuzz_corrupted_files_never_return_altered_text(tmp_path: Path):
    import random
    rng = random.Random(7)
    cache = ParseCache(tmp_path / "cache")
    text = "서버 백업은 매일 새벽에 실행됩니다. 휴가는 팀장 승인이 필요합니다."
    sha = "d" * 64
    served_wrong = 0
    for i in range(300):
        cache.put(sha, ".md", text)
        target = rng.choice([cache.mirror_path(sha, ".md"), cache._meta_path(sha, ".md")])
        data = bytearray(target.read_bytes())
        mode = rng.choice(["flip", "truncate", "empty", "append", "delete"])
        if mode == "flip" and data:
            data[rng.randrange(len(data))] ^= 1 << rng.randrange(8)
        elif mode == "truncate" and data:
            del data[rng.randrange(len(data)):]
        elif mode == "empty":
            data = bytearray()
        elif mode == "append":
            data += b"\xff\xfe junk"
        if mode == "delete":
            target.unlink()
        else:
            target.write_bytes(bytes(data))
        got = cache.get(sha, ".md")           # 예외 없이 None(= 다시 파싱) 이거나 원문 그대로여야 한다
        if got is not None and got != text:
            served_wrong += 1
    assert served_wrong == 0


def test_make_loader_flag_controls_caching(tmp_path: Path):
    from types import SimpleNamespace
    from intellectagent.document_ingest import make_loader
    root = tmp_path / "docs"; root.mkdir()
    cfg = SimpleNamespace(allowed_root=root, allowed_extensions=(".md",), max_file_bytes=1_000_000,
                          index_dir=tmp_path / "index" / "chroma", feature_flags={})
    assert isinstance(make_loader(cfg), SafeDocumentLoader)
    cfg.feature_flags = {"use_parse_cache": True}
    cached = make_loader(cfg)
    assert isinstance(cached, CachingDocumentLoader) and cached.cache.root == (tmp_path / "index" / "parse_cache").resolve()
    cfg.index_dir = root / "idx" / "chroma"                      # 캐시가 허용 루트 안으로 들어가는 설정은 거부
    with pytest.raises(ValueError):
        make_loader(cfg)

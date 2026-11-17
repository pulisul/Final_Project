import math
from pathlib import Path

import pytest

from intellectagent.grid_search import Candidate, filter_by_speed, run_grid, select_best
from intellectagent.hybrid import HybridRetriever, rrf_fuse
from intellectagent.lexical import BM25Index, tok_char_bigram, tok_word
from intellectagent.models import Chunk, EvalCase, SearchHit
from intellectagent import retrieval_metrics as M


def H(cid, score, doc=None):
    return SearchHit(cid, doc or cid, f"text-{cid}", score)


# ---- 지표 -----------------------------------------------------------------
def test_metrics_hand_computed():
    gt, pred = [["a"], ["b"]], ["a", "x", "b"]
    assert M.retrieval_recall(gt, pred) == pytest.approx(1.0)
    assert M.retrieval_precision(gt, pred) == pytest.approx(2 / 3)
    assert M.retrieval_f1(gt, pred) == pytest.approx(0.8)
    assert M.retrieval_mrr(gt, pred) == pytest.approx((1 + 1 / 3) / 2)
    assert M.retrieval_map(gt, pred) == pytest.approx((1 + 1 / 3) / 2)
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert M.retrieval_ndcg(gt, pred) == pytest.approx(dcg / idcg)


def test_metrics_empty_prediction_is_zero_not_skipped():
    for name, fn in M.METRICS.items():
        assert fn([["a"]], []) == 0.0, name


def test_metrics_group_means_any_member_counts():
    assert M.retrieval_recall([["a", "a2"]], ["a2"]) == 1.0


def test_unknown_metric_rejected():
    with pytest.raises(KeyError):
        M.score_case([["a"]], ["a"], ["nope"])


# ---- 토크나이저 / BM25 ------------------------------------------------------
def _chunks(texts):
    return [Chunk(f"c{i}", f"d{i}", t, "x", i) for i, t in enumerate(texts)]


def test_bigram_absorbs_korean_particles_where_word_tokenizer_fails():
    chunks = _chunks(["지도는 배송 경로를 보여 준다", "사과는 과일이다", "자동차는 이동수단이다"])
    q = "지도를 보여줘"
    assert BM25Index(chunks, tok_word).search(q, 3) == []  # 어절이 달라서(지도를 != 지도는) 못 찾는다
    hits = BM25Index(chunks, tok_char_bigram).search(q, 3)
    assert hits and hits[0].doc_id == "d0"


def test_bm25_does_not_return_zero_score_chunks():
    idx = BM25Index(_chunks(["alpha beta", "gamma delta", "eps zeta", "eta theta"]), tok_word)
    assert [h.doc_id for h in idx.search("alpha", 5)] == ["d0"]


def test_okapi_idf_is_zero_on_two_doc_corpus_and_lucene_variant_fixes_it():
    chunks = _chunks(["alpha beta", "gamma delta"])  # 샘플 코퍼스처럼 문서가 2건
    assert BM25Index(chunks, tok_word).search("alpha", 5) == []  # 알려진 Okapi 성질: 절반 문서에 나오면 idf=0
    assert [h.doc_id for h in BM25Index(chunks, tok_word, idf_variant="lucene").search("alpha", 5)] == ["d0"]
    with pytest.raises(ValueError):
        BM25Index(chunks, tok_word, idf_variant="x")


def test_bm25_matches_rank_bm25():
    rank_bm25 = pytest.importorskip("rank_bm25")
    texts = ["a b c a", "b c d", "e f a", "a a a b", "x y z"]
    mine = BM25Index(_chunks(texts), tok_word)
    ref = rank_bm25.BM25Okapi([tok_word(t) for t in texts])
    for q in ["a", "a b", "z y", "q"]:
        for x, y in zip(mine.scores(q), ref.get_scores(tok_word(q))):
            assert x == pytest.approx(y, abs=1e-12)


def test_bm25_empty_corpus():
    assert BM25Index([], tok_word).search("a", 3) == []


# ---- RRF ------------------------------------------------------------------
def test_rrf_hand_computed_order():
    A = [H("a", 3), H("b", 2), H("c", 1)]
    B = [H("c", 9), H("a", 8), H("d", 7)]
    out = rrf_fuse([A, B], top_k=4, k=60)
    exp = {"a": 1 / 61 + 1 / 62, "c": 1 / 63 + 1 / 61, "b": 1 / 62, "d": 1 / 63}
    assert [h.chunk_id for h in out] == ["a", "c", "b", "d"]
    for h in out:
        assert h.score == pytest.approx(exp[h.chunk_id])


def test_rrf_ties_share_min_rank():
    A = [H("a", 1.0), H("b", 1.0), H("c", 0.5)]
    out = {h.chunk_id: h.score for h in rrf_fuse([A, [H("z", 1.0)]], top_k=5, k=60)}
    assert out["a"] == pytest.approx(1 / 61) and out["b"] == pytest.approx(1 / 61) and out["c"] == pytest.approx(1 / 63)


def test_rrf_input_validation():
    with pytest.raises(ValueError):
        rrf_fuse([[H("a", 1)]], top_k=1)
    with pytest.raises(ValueError):
        rrf_fuse([[H("a", 1)], [H("b", 1)]], top_k=0)


class _Boom:
    def search(self, q, k):
        raise RuntimeError("down")


class _Fixed:
    def __init__(self, hits): self.hits = hits
    def search(self, q, k): return self.hits[:k]


def test_hybrid_degrades_to_single_side_and_to_empty():
    assert [h.chunk_id for h in HybridRetriever(_Boom(), _Fixed([H("a", 1)])).search("q", 3)] == ["a"]
    assert HybridRetriever(_Boom(), _Boom()).search("q", 3) == []


# ---- 격자 탐색 --------------------------------------------------------------
CASES = [EvalCase("q1", "q1", ("d1",)), EvalCase("q2", "q2", ("d2",))]


def _cand(name, mapping):
    return Candidate(name, lambda q, k: [H(f"c-{d}", 1, d) for d in mapping[q]][:k])


def test_run_grid_selects_best_and_writes_files(tmp_path: Path):
    good = _cand("good", {"q1": ["d1"], "q2": ["d2"]})
    bad = _cand("bad", {"q1": ["x"], "q2": []})
    res = run_grid([bad, good], CASES, top_k=3, out_dir=tmp_path)
    assert res["best"] == "good"
    assert next(r for r in res["rows"] if r["candidate"] == "bad")["retrieval_recall"] == 0.0
    assert (tmp_path / "grid_result.json").exists() and (tmp_path / "grid_result.csv").exists()


def test_speed_filter_keeps_everything_when_all_too_slow():
    rows = [{"candidate": "a", "latency_ms_mean": 50.0}, {"candidate": "b", "latency_ms_mean": 60.0}]
    assert filter_by_speed(rows, 10.0) == rows
    assert filter_by_speed(rows, 55.0) == [rows[0]]


def test_rank_strategy_differs_from_mean_when_metrics_disagree():
    rows = [
        {"candidate": "x", "m1": 1.0, "m2": 0.0, "m3": 0.0},
        {"candidate": "y", "m1": 0.0, "m2": 0.6, "m3": 0.6},
    ]
    ms = ("m1", "m2", "m3")
    assert select_best(rows, ms, "mean")["candidate"] == "y"
    assert select_best(rows, ms, "rank")["candidate"] == "y"
    with pytest.raises(ValueError):
        select_best(rows, ms, "nope")


def test_rank_strategy_uses_average_rank_for_ties_like_autorag():
    # 손계산(원본 pandas rank 기본값=평균 순위): c0=1/3+1+1/3=1.667, c1=1/4+1/3.5+1=1.536, c3=1/1.5+1/2+1/3=1.5, c2=1.286
    ms = ["a", "b", "c"]
    rows = [{"candidate": "c0", "a": .75, "b": 1.0, "c": 0.0}, {"candidate": "c1", "a": .25, "b": .25, "c": 1.0},
            {"candidate": "c2", "a": 1.0, "b": .25, "c": 0.0}, {"candidate": "c3", "a": 1.0, "b": .5, "c": 0.0}]
    assert select_best(rows, ms, "rank")["candidate"] == "c0"


def test_normalize_mean_strategy_and_constant_metric():
    ms = ["a", "b"]
    rows = [{"candidate": "x", "a": 0.2, "b": 0.5}, {"candidate": "y", "a": 0.6, "b": 0.5}, {"candidate": "z", "a": 0.4, "b": 0.5}]
    assert select_best(rows, ms, "normalize_mean")["candidate"] == "y"  # b 는 전원 동일 -> 합에서 제외(0)
    with pytest.raises(ValueError):
        select_best(rows, ms, "nope")

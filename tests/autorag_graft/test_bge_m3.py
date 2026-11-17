import hashlib
import math
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from intellectagent import bge_m3 as m3
from intellectagent.chunk_grid import RetrieverSpec, build_candidates
from intellectagent.chunk_strategies import ChunkStrategyConfig
from intellectagent.models import Chunk, Document


class RandEnc:
    """텍스트 해시로 정해지는 임의의 dense/sparse/colbert. 호출 횟수를 센다."""
    name = "rand"

    def __init__(self, dim=8):
        self.dim, self.calls, self.colbert_calls = dim, 0, 0

    def _rng(self, text):
        return random.Random(int(hashlib.sha256(text.encode()).hexdigest()[:12], 16))

    def encode(self, texts, *, colbert=False):
        out = []
        for t in texts:
            self.calls += 1
            self.colbert_calls += bool(colbert)
            r = self._rng(t)
            d = [r.gauss(0, 1) for _ in range(self.dim)]
            n = math.sqrt(sum(x * x for x in d)); d = [x / n for x in d]
            sp = {f"t{r.randrange(12)}": r.random() for _ in range(r.randrange(1, 6))}
            cv = None
            if colbert:
                cv = np.array([[r.gauss(0, 1) for _ in range(self.dim)] for _ in range(r.randrange(2, 7))], dtype=np.float32)
                cv /= np.linalg.norm(cv, axis=1, keepdims=True)
            out.append(m3.M3Output(d, sp, cv))
        return out


def chunks(n=30):
    return [Chunk(f"c{i}", f"d{i % 4}", f"본문 번호 {i} " * 3, "length", i) for i in range(n)]


# FlagEmbedding 1.3.5 의 공식을 파이썬 반복문으로 따로 구현한 기준값
def ref_sparse(q, d):
    s = 0
    for t, w in q.items():
        if t in d:
            s += w * d[t]
    return s


def ref_colbert(q, d):
    tot = 0.0
    for qi in q:
        tot += max(sum(float(a) * float(b) for a, b in zip(qi, dj)) for dj in d)
    return tot / len(q)


def test_score_functions_match_flagembedding_definitions():
    assert m3.sparse_score({"a": 0.5, "b": 3.0, "z": 9}, {"a": 4.0, "b": 0.7}) == pytest.approx(0.5 * 4 + 3.0 * 0.7)
    assert m3.sparse_score({"a": 1.0}, {"b": 1.0}) == 0.0
    q = np.array([[1, 0], [0, 1]], np.float32); d = np.array([[1, 0], [0.6, 0.8], [0, -1]], np.float32)
    assert m3.colbert_score(q, d) == pytest.approx(ref_colbert(q, d)) == pytest.approx((1.0 + 0.8) / 2)
    assert m3.colbert_score(np.zeros((0, 2), np.float32), d) == 0.0
    assert m3.combine(0.5, 0.2, 0.9, (0.4, 0.2, 0.4)) == pytest.approx((0.4 * 0.5 + 0.2 * 0.2 + 0.4 * 0.9) / 1.0)
    assert m3.combine(0.5, 0.2, 0.9, (1, 1, 1)) == pytest.approx((0.5 + 0.2 + 0.9) / 3)
    assert m3.combine(0.5, 0.2, None, (0.4, 0.2, 0.4)) == pytest.approx((0.4 * 0.5 + 0.2 * 0.2) / 0.6)


@pytest.mark.parametrize("weights", [(0.4, 0.2, 0.4), (1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.2, 0.2, 0.6)])
def test_full_candidate_ranking_equals_bruteforce_reference(weights):
    cs, enc = chunks(), RandEnc()
    r = m3.M3Retriever(lambda: cs, enc, None, weights, colbert_rerank=True, candidate_k=len(cs))
    for query in ("번호 7 본문", "아무 질문", "t3 t5"):
        qo = enc.encode([query], colbert=True)[0]
        ref = []
        for c in cs:
            do = enc.encode([c.text], colbert=True)[0]
            dense = sum(a * b for a, b in zip(qo.dense, do.dense))
            ref.append((m3.combine(dense, ref_sparse(qo.sparse, do.sparse), ref_colbert(qo.colbert, do.colbert), weights), c.chunk_id))
        ref.sort(key=lambda x: (-x[0], int(x[1][1:])))
        hits = r.search(query, 5)
        assert [h.chunk_id for h in hits] == [cid for _, cid in ref[:5]]
        assert [h.score for h in hits] == pytest.approx([s for s, _ in ref[:5]], abs=1e-5)


def test_colbert_only_rescored_for_candidates_and_dense_sparse_only_mode_skips_it():
    cs, enc = chunks(), RandEnc()
    r = m3.M3Retriever(lambda: cs, enc, None, colbert_rerank=True, candidate_k=6)
    r.warm(); before = enc.colbert_calls
    hits = r.search("번호", 3)
    assert enc.colbert_calls - before == 1 + 6 and all("m3_colbert" in h.metadata for h in hits)   # 질의 1 + 후보 6
    r.search("번호", 3)
    assert enc.colbert_calls - before == 1 + 6 + 1                                             # 같은 후보는 LRU 재사용
    enc2 = RandEnc(); r2 = m3.M3Retriever(lambda: cs, enc2, None, colbert_rerank=False)
    assert all("m3_colbert" not in h.metadata for h in r2.search("번호", 3)) and enc2.colbert_calls == 0


def test_cache_encodes_only_new_or_changed_text(tmp_path):
    cache = m3.M3Cache(tmp_path / "m3.sqlite")
    cs = chunks(10); enc = RandEnc()
    r = m3.M3Retriever(lambda: cs, enc, cache, colbert_rerank=False); r.warm()
    assert r.stats["encoded"] == 10
    enc2 = RandEnc(); r2 = m3.M3Retriever(lambda: cs, enc2, cache, colbert_rerank=False); r2.warm()
    assert enc2.calls == 0 and r2.stats["cache_hits"] == 10
    cs.append(Chunk("new", "d9", "새로 들어온 문서 본문", "length", 0)); r2.warm()
    assert enc2.calls == 1
    cs[0] = Chunk("c0", "d0", cs[0].text, "length", 0, {"context": "[새 머리말]"}); r2.warm()      # 머리말이 바뀌면 다시 계산
    assert enc2.calls == 2
    cs.pop(); r2.warm()
    assert len(r2._chunks) == 10 and enc2.calls == 2 and len(r2._mem) == 10      # 삭제된 청크는 메모리에서도 빠진다


def test_corrupt_cache_rows_are_recomputed_not_trusted(tmp_path):
    cache = m3.M3Cache(tmp_path / "m3.sqlite"); cs = chunks(3); enc = RandEnc()
    m3.M3Retriever(lambda: cs, enc, cache, colbert_rerank=False).warm()
    import sqlite3
    with sqlite3.connect(cache.path) as c:
        c.execute("UPDATE m3 SET dim = dim + 1"); 
    enc2 = RandEnc(); r = m3.M3Retriever(lambda: cs, enc2, cache, colbert_rerank=False); r.warm()
    assert enc2.calls == 3 and r.search("번호", 1)


def test_hits_keep_original_text_and_expose_dense_as_gate_score():
    ctx = [Chunk("a", "d", "인증서를 설치해야 한다", "length", 0, {"context": "[vpn.md > 접속 절차]"})]
    r = m3.M3Retriever(lambda: ctx, m3.HashM3Encoder(), None, colbert_rerank=True)
    h = r.search("접속 절차 인증서", 1)[0]
    assert h.text == "인증서를 설치해야 한다" and h.metadata["gate_score"] == h.metadata["m3_dense"] and h.metadata["context"] == "[vpn.md > 접속 절차]"
    assert h.metadata["m3_sparse"] > 0       # 질의의 '접속 절차' 는 머리말에만 있다: 색인 글(머리말+본문)로 sparse 가 잡힌다


def test_invalid_arguments_and_dimension_mismatch():
    with pytest.raises(ValueError):
        m3.M3Retriever(lambda: [], RandEnc(), None, (1, 1))
    with pytest.raises(ValueError):
        m3.M3Retriever(lambda: [], RandEnc(), None, (0, 0, 0))
    with pytest.raises(ValueError):
        m3.M3Retriever(lambda: [], RandEnc(), None, (-1, 1, 1))
    assert m3.M3Retriever(lambda: [], RandEnc()).search("x", 3) == []
    cs = chunks(3); r = m3.M3Retriever(lambda: cs, RandEnc(8), None, colbert_rerank=False); r.warm(); r.encoder = RandEnc(5)
    with pytest.raises(ValueError):
        r.search("질문", 1)


def test_flag_encoder_wrapper_converts_outputs_and_passes_arguments(monkeypatch):
    seen = {}

    class FakeModel:
        def __init__(self, name, use_fp16, devices):
            seen["init"] = (name, use_fp16, devices)

        def encode(self, texts, **kw):
            seen["kw"] = kw
            n = len(texts)
            return {"dense_vecs": np.ones((n, 3), np.float16) / math.sqrt(3), "lexical_weights": [{"101": np.float16(0.5)} for _ in texts],
                    "colbert_vecs": [np.ones((4, 3), np.float32) for _ in texts] if kw["return_colbert_vecs"] else None}

    mod = types.ModuleType("FlagEmbedding"); mod.BGEM3FlagModel = FakeModel
    monkeypatch.setitem(sys.modules, "FlagEmbedding", mod)
    enc = m3.FlagM3Encoder("BAAI/bge-m3", device="cpu", batch_size=4, max_length=256)
    assert seen["init"] == ("BAAI/bge-m3", False, "cpu")           # CPU 에서는 fp16 을 끈다
    o = enc.encode(["a", "b"], colbert=True)
    assert seen["kw"]["return_dense"] and seen["kw"]["return_sparse"] and seen["kw"]["return_colbert_vecs"] and seen["kw"]["max_length"] == 256
    assert o[0].sparse == {"101": 0.5} and isinstance(o[0].dense[0], float) and o[0].colbert.shape == (4, 3)
    assert enc.encode(["a"])[0].colbert is None and enc.encode([]) == []


def test_flag_encoder_without_package_gives_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "FlagEmbedding", None)
    with pytest.raises(RuntimeError, match="pip install FlagEmbedding"):
        m3.FlagM3Encoder()


def test_grid_candidate_and_spec_names_and_search():
    d = Document("d1", Path("a.md"), "서버 백업 지침: 매일 새벽에 수행한다. " * 10, "h", {"name": "a.md"})
    specs = [RetrieverSpec("bge_m3"), RetrieverSpec("bge_m3", m3_weights=(1, 0, 0)), RetrieverSpec("bge_m3", m3_colbert=False)]
    assert [s.name for s in specs] == ["bge_m3+dsc", "bge_m3+d-w1:0:0", "bge_m3+ds"]
    cands = build_candidates([d], [ChunkStrategyConfig("length", 60, 10)], specs, m3_encoder=m3.HashM3Encoder())
    assert len(cands) == 3 and all(c.search("백업", 2) for c in cands)


def test_non_colbert_score_is_normalized_dense_sparse_mix_and_dense_is_unit_scaled():
    class Raw(RandEnc):
        def encode(self, texts, *, colbert=False):
            o = super().encode(texts, colbert=colbert)
            for x in o:
                x.dense = [v * 5.0 for v in x.dense]          # 정규화되지 않은 인코더
            return o
    cs, enc = chunks(8), Raw()
    r = m3.M3Retriever(lambda: cs, enc, None, (0.4, 0.2, 0.4), colbert_rerank=False)
    q = enc.encode(["번호 3"])[0]
    h = r.search("번호 3", 8)
    for x in h:
        d = enc.encode([next(c.text for c in cs if c.chunk_id == x.chunk_id)])[0]
        dense = sum(a * b for a, b in zip(q.dense, d.dense)) / (np.linalg.norm(q.dense) * np.linalg.norm(d.dense))   # 코사인
        assert x.metadata["m3_dense"] == pytest.approx(dense, abs=1e-5) and -1.0001 <= x.metadata["m3_dense"] <= 1.0001
        assert x.score == pytest.approx(m3.combine(dense, ref_sparse(q.sparse, d.sparse), None, (0.4, 0.2, 0.4)), abs=1e-5)


def test_colbert_lru_evicts_least_recently_used():
    cs, enc = chunks(6), RandEnc()
    r = m3.M3Retriever(lambda: cs, enc, None, colbert_rerank=True, candidate_k=3, colbert_lru=3)
    r.warm()
    r._doc_colbert([0, 1, 2]); r._doc_colbert([0]); r._doc_colbert([3])       # 0 을 다시 썼으니 가장 오래 안 쓴 것은 1
    kept = set(r._colbert)
    assert kept == {r._keys[0], r._keys[2], r._keys[3]} and len(kept) == 3

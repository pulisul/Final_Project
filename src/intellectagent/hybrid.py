"""하이브리드 검색: 여러 검색 결과를 RRF 로 융합한다 (AutoRAG ``hybrid_rrf`` 와 같은 정의).

RRF 점수 = sum( 1 / (rank_i + k) ),  rank_i 는 i 번째 검색 결과에서의 순위(동점은 min 순위),
해당 검색에 없으면 그 항은 0. k 기본값 60 (AutoRAG 의 ``weight``).
점수 스케일이 서로 다른 벡터(코사인)와 BM25 를 정규화 없이 합칠 수 있는 것이 장점이다.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

from .models import SearchHit


def _ranks(hits: Sequence[SearchHit]) -> dict[str, int]:
    """chunk_id -> 순위. 점수가 같으면 같은 순위(min), 순위 = 1 + 자신보다 점수가 높은 항목 수."""
    ranks: dict[str, int] = {}
    for h in hits:
        if h.chunk_id in ranks:
            continue  # 같은 청크가 한 목록에 두 번 있으면 첫 번째(더 높은) 위치만
        ranks[h.chunk_id] = 1 + sum(1 for o in hits if o.score > h.score)
    return ranks


def rrf_fuse(rankings: Sequence[Sequence[SearchHit]], top_k: int, k: int = 60) -> list[SearchHit]:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if k <= 0:
        raise ValueError("k must be > 0")
    if len(rankings) < 2:
        raise ValueError("RRF needs at least two rankings")
    fused: dict[str, float] = {}
    proto: dict[str, SearchHit] = {}
    for hits in rankings:
        for cid, r in _ranks(hits).items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (r + k)
        for h in hits:
            proto.setdefault(h.chunk_id, h)
    order = sorted(fused, key=lambda c: (-fused[c], c))[:top_k]
    return [SearchHit(c, proto[c].doc_id, proto[c].text, float(fused[c]), dict(proto[c].metadata)) for c in order]


# ---- CC (convex combination) : AutoRAG ``hybrid_cc`` ------------------------------------------------------------
# 각 검색의 점수를 정규화한 뒤 weight*semantic + (1-weight)*lexical 로 합친다. 한쪽에 없는 문서는 그 항이 0.
# 원본(numpy/pandas)에서 0/0 -> NaN -> fillna(0) 으로 처리되는 퇴화 입력(점수가 전부 같음 등)은 여기서 전부 0.0 이 된다.

def _norm_mm(scores: Sequence[float], fixed_min: float = 0.0) -> list[float]:
    lo, hi = min(scores), max(scores)
    d = hi - lo
    return [0.0] * len(scores) if d == 0 else [(x - lo) / d for x in scores]


def _norm_tmm(scores: Sequence[float], fixed_min: float) -> list[float]:
    d = max(scores) - fixed_min
    return [0.0] * len(scores) if d == 0 else [(x - fixed_min) / d for x in scores]


def _mean_std(scores: Sequence[float]) -> tuple[float, float]:
    m = sum(scores) / len(scores)
    return m, math.sqrt(sum((x - m) ** 2 for x in scores) / len(scores))  # numpy.std (ddof=0)


def _norm_z(scores: Sequence[float], fixed_min: float = 0.0) -> list[float]:
    m, sd = _mean_std(scores)
    return [0.0] * len(scores) if sd == 0 else [(x - m) / sd for x in scores]


def _norm_dbsf(scores: Sequence[float], fixed_min: float = 0.0) -> list[float]:
    m, sd = _mean_std(scores)
    lo, hi = m - 3 * sd, m + 3 * sd
    return [0.0] * len(scores) if hi == lo else [(x - lo) / (hi - lo) for x in scores]


NORMALIZERS: dict[str, Callable[..., list[float]]] = {"mm": _norm_mm, "tmm": _norm_tmm, "z": _norm_z, "dbsf": _norm_dbsf}


def cc_fuse(semantic: Sequence[SearchHit], lexical: Sequence[SearchHit], top_k: int, weight: float,
            normalize_method: str = "mm", semantic_min: float = -1.0, lexical_min: float = 0.0) -> list[SearchHit]:
    """weight 는 semantic 쪽 가중치(1.0 이면 semantic 만). 동점은 먼저 나온 문서(semantic 목록 순서 우선)가 앞선다."""
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    if normalize_method not in NORMALIZERS:
        raise ValueError(f"unknown normalize_method {normalize_method!r} ({'|'.join(NORMALIZERS)})")
    norm = NORMALIZERS[normalize_method]
    sem = dict(zip((h.chunk_id for h in semantic), norm([h.score for h in semantic], semantic_min))) if semantic else {}
    lex = dict(zip((h.chunk_id for h in lexical), norm([h.score for h in lexical], lexical_min))) if lexical else {}
    order = list(dict.fromkeys([*sem, *lex]))  # 첫 등장 순서
    fused = {c: sem.get(c, 0.0) * weight + lex.get(c, 0.0) * (1.0 - weight) for c in order}
    proto: dict[str, SearchHit] = {}
    for h in [*semantic, *lexical]:
        proto.setdefault(h.chunk_id, h)
    top = sorted(order, key=lambda c: -fused[c])[:top_k]  # sorted 는 안정 정렬
    return [SearchHit(c, proto[c].doc_id, proto[c].text, float(fused[c]), dict(proto[c].metadata)) for c in top]


class HybridRetriever:
    """벡터 검색 + BM25 를 융합(``fusion="rrf"|"cc"``). 한쪽이 실패해도 다른 쪽 결과로 계속 동작한다(기존 Retriever 와 같은 정신)."""

    def __init__(self, vector_retriever, bm25, rrf_k: int = 60, candidate_k: int = 20,
                 fusion: str = "rrf", cc_weight: float = 0.5, cc_normalize: str = "mm"):
        if fusion not in ("rrf", "cc"):
            raise ValueError("fusion must be 'rrf' or 'cc'")
        self.vector = vector_retriever
        self.bm25 = bm25
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.fusion = fusion
        self.cc_weight = cc_weight
        self.cc_normalize = cc_normalize

    def search(self, query: str, top_k: int, candidate_k: int | None = None) -> list[SearchHit]:
        ck = max(candidate_k or self.candidate_k, top_k)
        vec: list[SearchHit] = []
        lex: list[SearchHit] = []
        try:
            vec = list(self.vector.search(query, ck))
        except Exception:
            pass
        try:
            lex = list(self.bm25.search(query, ck))
        except Exception:
            pass
        if not vec and not lex:
            return []
        if not vec or not lex:
            return (vec or lex)[:top_k]
        if self.fusion == "cc":
            return cc_fuse(vec, lex, top_k=top_k, weight=self.cc_weight, normalize_method=self.cc_normalize)
        return rrf_fuse([vec, lex], top_k=top_k, k=self.rrf_k)


# ---- 파이프라인 연결용 -------------------------------------------------------------------------------------------
import hashlib
import threading


def store_chunks(store) -> list:
    """벡터 저장소에 들어 있는 모든 청크(BM25 색인의 원본). InMemory / Chroma 지원."""
    from .models import Chunk

    rows = getattr(store, "rows", None)
    if isinstance(rows, dict):
        return [c for c, _ in rows.values()]
    coll = getattr(store, "collection", None)
    if coll is None:
        raise TypeError(f"청크를 읽을 수 없는 저장소: {type(store).__name__}")
    r = coll.get(include=["documents", "metadatas"])
    out = []
    for cid, text, meta in zip(r.get("ids") or [], r.get("documents") or [], r.get("metadatas") or [], strict=False):
        meta = dict(meta or {})
        out.append(Chunk(cid, str(meta.pop("doc_id", "")), text or "", str(meta.pop("strategy", "")),
                         int(meta.pop("ordinal", 0) or 0), meta))
    return out


class StoreBM25:
    """저장소 내용이 바뀌면(문서 추가/삭제) 다음 검색 때 BM25 색인을 다시 만든다. 청크 ID 에 본문 해시가 들어 있어 ID 집합이 곧 내용 지문이다."""

    def __init__(self, store, tokenizer: str = "char_bigram", idf_variant: str = "lucene"):
        from .lexical import TOKENIZERS

        if tokenizer not in TOKENIZERS:
            raise ValueError(f"unknown tokenizer {tokenizer!r} ({'|'.join(TOKENIZERS)})")
        self.store, self.tokenizer, self.idf_variant = store, tokenizer, idf_variant
        self._index = None
        self._sig = None
        self._lock = threading.Lock()

    def _signature(self, chunks) -> str:
        h = hashlib.sha256()
        for cid in sorted(c.chunk_id for c in chunks):
            h.update(cid.encode())
            h.update(b"\0")
        return h.hexdigest()

    def search(self, query: str, top_k: int, candidate_k: int | None = None) -> list[SearchHit]:
        from .lexical import TOKENIZERS, BM25Index

        with self._lock:
            chunks = store_chunks(self.store)
            sig = self._signature(chunks)
            if sig != self._sig:
                self._index = BM25Index(chunks, TOKENIZERS[self.tokenizer], idf_variant=self.idf_variant) if chunks else None
                self._sig = sig
            index = self._index
        return index.search(query, candidate_k or top_k)[:top_k] if index is not None else []


class GatedHybridRetriever:
    """HybridRetriever + (선택) 리랭커. 융합 점수는 척도가 달라(RRF≈0.03, mm 은 항상 1.0) 기존 ``fallback_min_score`` 로 쓸 수 없다.
    리랭커가 없으면 각 결과에 ``gate_score`` = 벡터 코사인 원점수(벡터 목록에 없던 항목은 BM25 가 맞춘 것이므로 None)를 달고,
    CoreRAG 는 그 최댓값으로 '근거 없음'을 판정한다. 리랭커가 있으면 기존처럼 리랭커 점수가 판정 기준이다."""

    def __init__(self, hybrid: HybridRetriever, reranker=None):
        self.hybrid, self.reranker = hybrid, reranker

    def search(self, query: str, top_k: int, candidate_k: int | None = None) -> list[SearchHit]:
        ck = max(candidate_k or self.hybrid.candidate_k, top_k)
        vec_scores: dict[str, float] = {}
        try:
            vec_scores = {h.chunk_id: h.score for h in self.hybrid.vector.search(query, ck)}
        except Exception:
            pass
        hits = self.hybrid.search(query, ck if self.reranker is not None else top_k, ck)
        if self.reranker is not None:
            try:
                hits = list(self.reranker.rerank(query, hits))
            except Exception:
                pass
            return hits[:top_k]
        return [SearchHit(h.chunk_id, h.doc_id, h.text, h.score, {**h.metadata, "gate_score": vec_scores.get(h.chunk_id)}) for h in hits]

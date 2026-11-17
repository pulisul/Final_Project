"""BGE-M3 다중 표현 검색: 한 모델이 내놓는 dense / sparse(어휘 가중치) / ColBERT(토큰별 벡터) 를 함께 쓴다.

점수 공식은 FlagEmbedding 1.3.5 (``M3Embedder``) 소스와 같다.
- dense   : 정규화된 벡터의 내적(= 코사인)
- sparse  : 질의와 문서가 공유하는 토큰의 (질의 가중치 x 문서 가중치) 합
- colbert : 질의의 각 토큰에 대해 문서 토큰과의 최대 내적을 구해 질의 토큰 수로 평균
- 합산    : (w_d*dense + w_s*sparse + w_c*colbert) / (w_d + w_s + w_c).  모델 카드 권장 비율 0.4 : 0.2 : 0.4

실용 설계
- 1단계로 전체 청크를 dense+sparse 로 훑어 후보(candidate_k)를 고르고, 2단계에서 *후보만* ColBERT 로 다시 채점한다.
  ColBERT 벡터를 청크마다 저장하면 용량이 폭증한다(청크 3천 개 x 토큰 200 x 1024 x 2바이트 ≈ 1.2 GB 이상). 후보의 ColBERT 는 질의 때 계산하고 작은 LRU 로 재사용한다.
- dense/sparse 는 SQLite 캐시에 저장한다. 키 = sha256(모델 이름 + 색인 글). 본문이나 문맥 머리말이 바뀌면 자동으로 다른 키가 되고, 문서가 추가되면 새 청크만 인코딩한다.
- 색인 글은 ``models.index_text`` (문맥 머리말 + 본문). 결과의 ``text`` 는 항상 원래 본문이다.
- 결과마다 ``gate_score`` = dense 코사인을 달아 기존 "근거 없음" 판정(fallback_min_score)이 같은 척도로 동작하게 한다.
- 실제 모델(FlagEmbedding)은 필요할 때만 import 한다. 테스트·배관 확인용 HashM3Encoder 는 의미 임베딩이 아니다.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from .models import Chunk, SearchHit, index_text

DEFAULT_WEIGHTS = (0.4, 0.2, 0.4)   # dense, sparse, colbert


@dataclass
class M3Output:
    dense: list[float]
    sparse: dict[str, float]
    colbert: np.ndarray | None = None   # [토큰 수, 차원]


class M3Encoder(Protocol):
    name: str

    def encode(self, texts: Sequence[str], *, colbert: bool = False) -> list[M3Output]: ...


class FlagM3Encoder:
    """FlagEmbedding 의 BGEM3FlagModel 래퍼 (pip install FlagEmbedding)."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None, use_fp16: bool | None = None,
                 batch_size: int = 16, max_length: int = 512):
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise RuntimeError("BGE-M3 다중 표현에는 FlagEmbedding 이 필요합니다: pip install FlagEmbedding") from exc
        self.name = f"flag:{model_name}"
        self.batch_size, self.max_length = batch_size, max_length
        fp16 = (device not in (None, "cpu")) if use_fp16 is None else use_fp16
        self.model = BGEM3FlagModel(model_name, use_fp16=fp16, devices=device)

    def encode(self, texts: Sequence[str], *, colbert: bool = False) -> list[M3Output]:
        if not texts:
            return []
        r = self.model.encode(list(texts), batch_size=self.batch_size, max_length=self.max_length,
                              return_dense=True, return_sparse=True, return_colbert_vecs=colbert)
        cols = r.get("colbert_vecs") if colbert else None
        return [M3Output([float(x) for x in r["dense_vecs"][i]], {str(k): float(v) for k, v in r["lexical_weights"][i].items()},
                         np.asarray(cols[i], dtype=np.float32) if cols is not None else None) for i in range(len(texts))]


class HashM3Encoder:
    """결정적인 가짜 인코더(의미 없음): 해시 기반 dense, 글자 bigram sparse, 토큰별 해시 벡터 ColBERT. 배관·테스트 전용."""

    name = "hash-m3"

    def __init__(self, dims: int = 64):
        self.dims = dims

    def _vec(self, token: str) -> np.ndarray:
        raw = hashlib.sha256(token.encode()).digest()
        v = np.frombuffer((raw * (self.dims // 32 + 1))[: self.dims], dtype=np.uint8).astype(np.float32) - 127.5
        return v / (np.linalg.norm(v) or 1.0)

    def encode(self, texts: Sequence[str], *, colbert: bool = False) -> list[M3Output]:
        from .lexical import tok_char_bigram

        out = []
        for t in texts:
            toks = tok_char_bigram(t)
            cnt: dict[str, int] = {}
            for x in toks:
                cnt[x] = cnt.get(x, 0) + 1
            n = max(len(toks), 1)
            sparse = {k: v / n for k, v in cnt.items()}
            vecs = np.stack([self._vec(x) for x in toks]) if toks else np.zeros((0, self.dims), np.float32)
            dense = vecs.sum(axis=0) if len(vecs) else np.zeros(self.dims, np.float32)
            dense = dense / (np.linalg.norm(dense) or 1.0)
            out.append(M3Output(dense.tolist(), sparse, vecs if colbert else None))
        return out


# ---- 점수 ---------------------------------------------------------------------------------------------------------
def sparse_score(q: dict[str, float], d: dict[str, float]) -> float:
    return float(sum(w * d[t] for t, w in q.items() if t in d))


def colbert_score(q: np.ndarray, d: np.ndarray) -> float:
    if q.shape[0] == 0 or d.shape[0] == 0:
        return 0.0
    return float((q @ d.T).max(axis=1).sum() / q.shape[0])


def combine(dense: float, sparse: float, colbert: float | None, weights: Sequence[float]) -> float:
    w = list(weights) if colbert is not None else [weights[0], weights[1]]
    vals = [dense, sparse] + ([colbert] if colbert is not None else [])
    total = sum(w)
    return sum(a * b for a, b in zip(w, vals)) / total if total > 0 else 0.0


# ---- 캐시 ---------------------------------------------------------------------------------------------------------
def cache_key(encoder_name: str, text: str) -> str:
    return hashlib.sha256(f"{encoder_name}\0{text}".encode()).hexdigest()


class M3Cache:
    """dense/sparse 를 SQLite 에 저장한다(원자적 커밋). 손상된 행은 없는 것으로 취급해 다시 계산한다."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS m3 (key TEXT PRIMARY KEY, dim INTEGER NOT NULL, dense BLOB NOT NULL, sparse TEXT NOT NULL)")

    def _conn(self):
        return sqlite3.connect(self.path, timeout=30)

    def get_many(self, keys: Sequence[str]) -> dict[str, tuple[np.ndarray, dict[str, float]]]:
        out: dict[str, tuple[np.ndarray, dict[str, float]]] = {}
        with self._lock, self._conn() as c:
            for i in range(0, len(keys), 500):
                part = list(keys[i:i + 500])
                rows = c.execute(f"SELECT key, dim, dense, sparse FROM m3 WHERE key IN ({','.join('?' * len(part))})", part).fetchall()
                for key, dim, blob, sp in rows:
                    try:
                        vec = np.frombuffer(blob, dtype=np.float32)
                        if vec.shape[0] != dim:
                            continue
                        out[key] = (vec.copy(), {str(k): float(v) for k, v in json.loads(sp).items()})
                    except Exception:
                        continue
        return out

    def put_many(self, items: dict[str, tuple[np.ndarray, dict[str, float]]]) -> None:
        if not items:
            return
        rows = [(k, int(v.shape[0]), np.asarray(v, dtype=np.float32).tobytes(), json.dumps(sp, ensure_ascii=False)) for k, (v, sp) in items.items()]
        with self._lock, self._conn() as c:
            c.executemany("INSERT OR REPLACE INTO m3 (key, dim, dense, sparse) VALUES (?,?,?,?)", rows)


# ---- 검색기 -------------------------------------------------------------------------------------------------------
class M3Retriever:
    """chunks_fn() 이 돌려주는 청크 전체를 색인한다. 목록이 바뀌면(문서 추가/삭제) 다음 검색 때 새 청크만 인코딩해 갱신한다."""

    def __init__(self, chunks_fn: Callable[[], Sequence[Chunk]], encoder: M3Encoder, cache: M3Cache | None = None,
                 weights: Sequence[float] = DEFAULT_WEIGHTS, colbert_rerank: bool = True, candidate_k: int = 20,
                 batch_size: int = 32, colbert_lru: int = 512, memory: dict | None = None):
        if len(weights) != 3 or any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("weights must be 3 non-negative numbers (dense, sparse, colbert) with positive sum")
        if candidate_k <= 0:
            raise ValueError("candidate_k must be > 0")
        self.chunks_fn, self.encoder, self.cache = chunks_fn, encoder, cache
        self.weights, self.colbert_rerank, self.candidate_k, self.batch_size = tuple(float(w) for w in weights), colbert_rerank, candidate_k, batch_size
        self._lock = threading.Lock()
        self._mem: dict[str, tuple[np.ndarray, dict[str, float]]] = memory if memory is not None else {}   # 같은 청크 집합을 쓰는 여러 검색기가 인코딩 결과를 공유할 때 넘긴다
        self._sig: str | None = None
        self._chunks: list[Chunk] = []
        self._dense = np.zeros((0, 0), np.float32)
        self._sparse: list[dict[str, float]] = []
        self._inv: dict[str, list[tuple[int, float]]] = {}
        self._colbert: OrderedDict[str, np.ndarray] = OrderedDict()
        self._colbert_cap = colbert_lru
        self.stats = {"encoded": 0, "cache_hits": 0}

    @staticmethod
    def _unit(v: Sequence[float]) -> np.ndarray:
        a = np.asarray(v, dtype=np.float32)
        n = float(np.linalg.norm(a))
        return a / n if n > 0 else a

    def warm(self) -> int:
        """전체 청크를 미리 인코딩한다(첫 질문이 느려지는 것을 피하려고 색인 직후 호출). 청크 수를 돌려준다."""
        with self._lock:
            self._refresh()
            return len(self._chunks)

    def _refresh(self) -> None:
        chunks = list(self.chunks_fn())
        keys = [cache_key(self.encoder.name, index_text(c)) for c in chunks]
        sig = hashlib.sha256("\0".join(sorted(keys)).encode()).hexdigest()
        if sig == self._sig:
            return
        missing = [k for k in dict.fromkeys(keys) if k not in self._mem]
        if missing and self.cache is not None:
            got = self.cache.get_many(missing)
            self.stats["cache_hits"] += len(got)
            self._mem.update(got)
            missing = [k for k in missing if k not in self._mem]
        if missing:
            text_of = dict(zip(keys, (index_text(c) for c in chunks)))
            fresh: dict[str, tuple[np.ndarray, dict[str, float]]] = {}
            for i in range(0, len(missing), self.batch_size):
                part = missing[i:i + self.batch_size]
                for k, o in zip(part, self.encoder.encode([text_of[k] for k in part], colbert=False), strict=True):
                    fresh[k] = (self._unit(o.dense), dict(o.sparse))
            self.stats["encoded"] += len(fresh)
            self._mem.update(fresh)
            if self.cache is not None:
                self.cache.put_many(fresh)
        live = set(keys)
        for k in [k for k in self._mem if k not in live]:     # 삭제된 청크의 메모리 정리
            del self._mem[k]
        self._chunks = chunks
        self._keys = keys
        dims = {self._mem[k][0].shape[0] for k in keys}
        if len(dims) > 1:
            raise ValueError(f"dense 차원이 섞여 있습니다: {sorted(dims)} (모델이 바뀌었으면 캐시를 지우세요)")
        self._dense = np.stack([self._mem[k][0] for k in keys]) if keys else np.zeros((0, 0), np.float32)
        self._sparse = [self._mem[k][1] for k in keys]
        inv: dict[str, list[tuple[int, float]]] = {}
        for i, sp in enumerate(self._sparse):
            for t, w in sp.items():
                inv.setdefault(t, []).append((i, w))
        self._inv = inv
        self._sig = sig

    def _doc_colbert(self, idx: Sequence[int]) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        need = []
        for i in idx:
            k = self._keys[i]
            if k in self._colbert:
                self._colbert.move_to_end(k)
                out[i] = self._colbert[k]
            else:
                need.append(i)
        for s in range(0, len(need), self.batch_size):
            part = need[s:s + self.batch_size]
            for i, o in zip(part, self.encoder.encode([index_text(self._chunks[i]) for i in part], colbert=True), strict=True):
                v = o.colbert if o.colbert is not None else np.zeros((0, 1), np.float32)
                out[i] = v
                self._colbert[self._keys[i]] = v
        while len(self._colbert) > self._colbert_cap:
            self._colbert.popitem(last=False)
        return out

    def search(self, query: str, top_k: int, candidate_k: int | None = None) -> list[SearchHit]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        with self._lock:
            self._refresh()
            n = len(self._chunks)
            if n == 0:
                return []
            use_colbert = self.colbert_rerank and self.weights[2] > 0
            q = self.encoder.encode([query], colbert=use_colbert)[0]
            qd = self._unit(q.dense)
            if qd.shape[0] != self._dense.shape[1]:
                raise ValueError(f"질의 dense 차원 {qd.shape[0]} != 색인 차원 {self._dense.shape[1]} (모델이 바뀌었으면 캐시를 지우세요)")
            dense = self._dense @ qd
            sparse = np.zeros(n, np.float64)
            for t, w in q.sparse.items():
                for i, dw in self._inv.get(t, ()):
                    sparse[i] += w * dw
            w0, w1, w2 = self.weights
            two = (w0 + w1) or 1.0
            first = (w0 * dense + w1 * sparse) / two
            order = sorted(range(n), key=lambda i: (-float(first[i]), i))
            col: dict[int, float] = {}
            if use_colbert and q.colbert is not None:
                cand = order[:max(candidate_k or self.candidate_k, top_k)]
                dcol = self._doc_colbert(cand)
                col = {i: colbert_score(q.colbert, dcol[i]) for i in cand}
                final = {i: combine(float(dense[i]), float(sparse[i]), col[i], self.weights) for i in cand}
                ranked = sorted(cand, key=lambda i: (-final[i], i))[:top_k]
            else:
                final = {i: float(first[i]) for i in order[:top_k]}
                ranked = order[:top_k]
            hits = []
            for i in ranked:
                c = self._chunks[i]
                md = {**c.metadata, "gate_score": float(dense[i]), "m3_dense": float(dense[i]), "m3_sparse": float(sparse[i])}
                if i in col:
                    md["m3_colbert"] = col[i]
                hits.append(SearchHit(c.chunk_id, c.doc_id, c.text, float(final[i]), md))
            return hits

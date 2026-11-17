"""청킹 설정 x 검색 방식 격자 (AutoRAG 의 "청킹을 바꿔 가며 검색 성능 비교" 흐름).

``grid_search.run_grid`` 의 후보(Candidate)를 만들어 준다. 후보 하나 = (청킹 설정, 검색 방식, 선택적 리랭커).
- 청킹 설정마다 청크/BM25/벡터 인덱스를 따로 만든다(청크 ID 가 달라지므로 인덱스를 공유할 수 없다).
- 청킹 설정끼리의 비교는 ``retrieval_token_*`` 지표(정답 근거 텍스트 기준)로 한다. ID 기반 지표는 청크 ID 가 달라 비교 불가.
  문서 단위 지표(``relevant_doc_ids``)는 어느 청크든 맞은 문서면 정답 처리하므로 청킹 차이에 둔감하다.
- 기본 임베더는 HashEmbedder(의미 임베딩 아님). 실제 비교는 ``embedder=SentenceTransformerEmbedder("BAAI/bge-m3")`` 로 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .chunk_strategies import ChunkStrategyConfig, chunk_with_strategy
from .embeddings import HashEmbedder
from .grid_search import Candidate
from .hybrid import HybridRetriever
from .lexical import TOKENIZERS, BM25Index
from .models import Chunk, Document, SearchHit, index_text
from .retrieval import Retriever
from .vectorstore import InMemoryVectorStore

KINDS = ("vector", "bm25", "hybrid_rrf", "hybrid_cc", "bge_m3")


@dataclass(frozen=True)
class RetrieverSpec:
    kind: str                      # vector | bm25 | hybrid_rrf | hybrid_cc
    tokenizer: str = "char_bigram"  # bm25 / hybrid 에서만 사용 (lexical.TOKENIZERS 의 이름)
    idf_variant: str = "lucene"
    rrf_k: int = 60
    cc_weight: float = 0.5          # semantic 쪽 가중치
    cc_normalize: str = "mm"
    candidate_k: int = 20
    reranker: str | None = None     # build_candidates(rerankers={...}) 의 키
    m3_weights: tuple = (0.4, 0.2, 0.4)   # bge_m3: dense, sparse, colbert 비율
    m3_colbert: bool = True               # bge_m3: 후보를 ColBERT 로 재채점할지

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r} ({'|'.join(KINDS)})")
        if self.kind not in ("vector", "bge_m3") and self.tokenizer not in TOKENIZERS:
            raise ValueError(f"unknown tokenizer {self.tokenizer!r} ({'|'.join(TOKENIZERS)})")

    @property
    def name(self) -> str:
        if self.kind == "bge_m3":
            on = [n for n, w in zip(("d", "s", "c"), self.m3_weights) if w > 0 and (n != "c" or self.m3_colbert)]
            return "bge_m3+" + "".join(on) + ("" if self.m3_weights == (0.4, 0.2, 0.4) else "-w" + ":".join(f"{w:g}" for w in self.m3_weights)) + (f"+rr-{self.reranker}" if self.reranker else "")
        parts = [self.kind]
        if self.kind != "vector":
            parts.append(self.tokenizer)
        if self.kind == "hybrid_cc":
            parts.append(f"w{self.cc_weight:g}-{self.cc_normalize}")
        if self.kind == "hybrid_rrf":
            parts.append(f"k{self.rrf_k}")
        if self.reranker:
            parts.append(f"rr-{self.reranker}")
        return "+".join(parts)


def chunk_label(cfg: ChunkStrategyConfig) -> str:
    base = f"semantic-{cfg.min_chars}-{cfg.max_chars}" if cfg.name == "semantic" else f"{cfg.name}-{cfg.size}-{cfg.overlap}"
    return base + ("+ctx" if cfg.context_header else "")


def _chunk_all(docs: Sequence[Document], cfg: ChunkStrategyConfig) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_with_strategy(d, cfg))
    return out


def build_candidates(docs: Sequence[Document], chunk_configs: Sequence[ChunkStrategyConfig], specs: Sequence[RetrieverSpec],
                     embedder=None, rerankers: Mapping[str, object] | None = None, name_prefix: str = "",
                     m3_encoder=None, m3_cache=None, gate: bool = False) -> list[Candidate]:
    """``gate=True`` 이면 하이브리드 후보를 운영 파이프라인과 같이 ``GatedHybridRetriever`` 로 감싸 hit 마다 ``gate_score``(벡터 코사인 원점수)를 단다.
    fallback 임계값 보정(real_env_eval)처럼 '운영과 같은 gate 값' 이 필요할 때만 켠다. 기본은 끔(기존 결과 불변)."""
    rerankers = dict(rerankers or {})
    missing = sorted({s.reranker for s in specs if s.reranker and s.reranker not in rerankers})
    if missing:
        raise KeyError(f"rerankers not provided: {missing}")
    labels = [chunk_label(c) for c in chunk_configs]
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate chunk configs")
    embedder = embedder or HashEmbedder()
    out: list[Candidate] = []
    for cfg, label in zip(chunk_configs, labels):
        chunks = _chunk_all(docs, cfg)
        if not chunks:
            raise ValueError(f"chunk config {label} produced no chunks")
        store = None
        if any(s.kind in ("vector", "hybrid_rrf", "hybrid_cc") for s in specs):
            store = InMemoryVectorStore()
            store.upsert(chunks, embedder.embed([index_text(c) for c in chunks]))
        vector = Retriever(embedder, store) if store is not None else None
        bm25_cache: dict[tuple[str, str], BM25Index] = {}
        m3_memory: dict = {}     # 이 청킹 설정의 모든 bge_m3 후보가 dense/sparse 인코딩을 공유한다(ablation 비용 1배)
        for spec in specs:
            bm25 = None
            if spec.kind == "bge_m3":
                from .bge_m3 import HashM3Encoder, M3Retriever

                inner = M3Retriever(lambda ch=chunks: ch, m3_encoder or HashM3Encoder(), m3_cache, spec.m3_weights,
                                    colbert_rerank=spec.m3_colbert, candidate_k=spec.candidate_k, memory=m3_memory)
                out.append(Candidate(f"{name_prefix}{label}|{spec.name}", _make_search(inner, spec, rerankers.get(spec.reranker or "")),
                                     {"chunk": label, "retriever": spec.name, "chunk_count": len(chunks)}))
                continue
            if spec.kind != "vector":
                key = (spec.tokenizer, spec.idf_variant)
                if key not in bm25_cache:
                    bm25_cache[key] = BM25Index(chunks, TOKENIZERS[spec.tokenizer], idf_variant=spec.idf_variant)
                bm25 = bm25_cache[key]
            if spec.kind == "vector":
                inner = vector
            elif spec.kind == "bm25":
                inner = bm25
            else:
                inner = HybridRetriever(vector, bm25, rrf_k=spec.rrf_k, candidate_k=spec.candidate_k,
                                        fusion="cc" if spec.kind == "hybrid_cc" else "rrf",
                                        cc_weight=spec.cc_weight, cc_normalize=spec.cc_normalize)
                if gate:
                    from .hybrid import GatedHybridRetriever
                    inner = GatedHybridRetriever(inner, None)
            out.append(Candidate(f"{name_prefix}{label}|{spec.name}", _make_search(inner, spec, rerankers.get(spec.reranker or "")),
                                 {"chunk": label, "retriever": spec.name, "chunk_count": len(chunks)}))
    return out


def _make_search(inner, spec: RetrieverSpec, reranker):
    def search(query: str, top_k: int) -> list[SearchHit]:
        if reranker is None:
            return list(inner.search(query, top_k))
        pool = list(inner.search(query, max(spec.candidate_k, top_k)))
        try:
            pool = list(reranker.rerank(query, pool))  # 리랭커 실패는 검색 실패로 만들지 않는다(기존 Retriever 와 같은 정책)
        except Exception:
            pass
        return pool[:top_k]
    return search

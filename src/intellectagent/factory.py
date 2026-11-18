from __future__ import annotations
import os
from .embeddings import HashEmbedder, SentenceTransformerEmbedder
from .vectorstore import InMemoryVectorStore, ChromaVectorStore
from .retrieval import Retriever, NoopReranker, CrossEncoderReranker
from .generation import ExtractiveGenerator, OllamaGenerator
from .pipeline import CoreRAG
from .extensions.factory import build_extension_registry
from .platform_runtime import detect_runtime_profile, configure_mac_mps_fallback


def build_components(cfg, runtime_profile: str | None = None):
    """Build one coherent runtime stack from one CoreConfig.

    Production components share the same runtime profile/device. Smoke-only components
    avoid importing torch/sentence-transformers entirely.
    """
    f = cfg.feature_flags
    needs_accelerator = bool(f.get("use_sentence_transformer") or f.get("use_reranker"))
    requested = runtime_profile or os.environ.get("RUNTIME_PROFILE") or getattr(cfg, "runtime_profile", "auto")
    profile = detect_runtime_profile(requested) if needs_accelerator else None
    if profile is not None:
        if not profile.pass_runtime:
            raise RuntimeError(f"runtime profile not ready: {profile.reason}")
        configure_mac_mps_fallback(profile)
    device = profile.torch_device if profile is not None else None

    embedder = SentenceTransformerEmbedder(cfg.embedding_model, device=device) if f.get("use_sentence_transformer") else HashEmbedder()
    store = ChromaVectorStore(str(cfg.index_dir), cfg.collection_name) if f.get("use_chroma") else InMemoryVectorStore()
    reranker = CrossEncoderReranker(cfg.reranker_model, device=device) if f.get("use_reranker") else NoopReranker()
    gen_opts = {k: f[n] for k, n in (("temperature", "ollama_temperature"), ("seed", "ollama_seed")) if f.get(n) is not None}
    generator = OllamaGenerator(cfg.ollama_model, citation_style=f.get("citation_style","index"), options=gen_opts or None) if f.get("use_ollama") else ExtractiveGenerator()
    mode = str(f.get("retrieval_mode", "vector"))
    if mode == "vector":
        retriever = Retriever(embedder, store, reranker)
    elif mode == "bge_m3":
        from pathlib import Path as _P
        from .bge_m3 import DEFAULT_WEIGHTS, FlagM3Encoder, HashM3Encoder, M3Cache, M3Retriever
        from .hybrid import store_chunks
        real = bool(f.get("use_sentence_transformer"))
        encoder = FlagM3Encoder(cfg.embedding_model, device=device) if real else HashM3Encoder()
        cache = M3Cache(_P(cfg.index_dir).parent / "m3_cache.sqlite") if real else None
        retriever = M3Retriever(lambda: store_chunks(store), encoder, cache, tuple(f.get("m3_weights", DEFAULT_WEIGHTS)),
                                colbert_rerank=bool(f.get("m3_colbert_rerank", True)), candidate_k=cfg.candidate_k or 20)
    elif mode in ("bm25", "hybrid_rrf", "hybrid_cc"):
        from .hybrid import GatedHybridRetriever, HybridRetriever, StoreBM25
        bm25 = StoreBM25(store, str(f.get("bm25_tokenizer", "char_bigram")))
        if mode == "bm25":
            retriever = bm25
        else:
            vector = Retriever(embedder, store, NoopReranker())
            hybrid = HybridRetriever(vector, bm25, candidate_k=cfg.candidate_k or max(20, cfg.top_k * 4),
                                     fusion="cc" if mode == "hybrid_cc" else "rrf",
                                     cc_weight=float(f.get("hybrid_cc_weight", 0.5)), cc_normalize=str(f.get("hybrid_cc_normalize", "mm")))
            retriever = GatedHybridRetriever(hybrid, reranker if f.get("use_reranker") else None)
    else:
        raise ValueError(f"unknown retrieval_mode {mode!r} (vector|bm25|hybrid_rrf|hybrid_cc|bge_m3)")
    if f.get("use_relevance_judge"):
        # 선택 기능(기본 꺼짐): LLM 이 청크별로 Yes/No 를 판정해 답이 못 되는 청크를 버린다. Ollama 필요.
        from .relevance_judge import JudgedRetriever, OllamaJudge, RelevanceJudge
        retriever = JudgedRetriever(retriever, RelevanceJudge(
            OllamaJudge(str(f.get("judge_model") or cfg.ollama_model), use_logprobs=bool(f.get("judge_logprobs", True))),
            n_std=float(f.get("judge_std_multiplier", 1.0)), max_candidates=int(f.get("judge_max_candidates", cfg.top_k))))
    extensions = build_extension_registry(cfg)
    return embedder, store, retriever, generator, CoreRAG(
        retriever,
        generator,
        cfg.top_k,
        cfg.fallback_min_score,
        extensions,
        candidate_k=cfg.candidate_k,
        runtime_profile=profile,
        grounding_check_numbers=bool(f.get("grounding_check_numbers", False)),
    )

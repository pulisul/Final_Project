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
    generator = OllamaGenerator(cfg.ollama_model) if f.get("use_ollama") else ExtractiveGenerator()
    retriever = Retriever(embedder, store, reranker)
    extensions = build_extension_registry(cfg)
    return embedder, store, retriever, generator, CoreRAG(
        retriever,
        generator,
        cfg.top_k,
        cfg.fallback_min_score,
        extensions,
        candidate_k=cfg.candidate_k,
        runtime_profile=profile,
    )

# Architecture

```text
SafeDocumentLoader -> Chunker -> Embedder -> VectorStore
                                      |          |
                                      +-> MetadataDB
Question -> Retriever -> [optional Reranker] -> CoreRAG -> Generator -> Grounding Gate -> Answer/Fallback
                                                        |
                                                        +-> deterministic Evaluation
                                                        +-> Evidence / Repro Manifest
Optional only after P0 gate: LangGraph / RAGAS / RL / Streamlit
```

## Failure containment
- Loader failure: record file only; continue other files.
- Embedding failure: checkpoint + failed list; no partial metadata commit.
- Reranker failure: planned production wrapper should fall back to base retrieval.
- LLM unsupported citation: fail closed to fallback.
- RAGAS API break: deterministic core metrics remain available.
- LangGraph break: CoreRAG direct API remains available.
- RL underperformance: policy rolls back to rule-based policy.

# Changelog

## GitHub-ready snapshot — 2026-09-30

- Based on `IntellectAgent_Weekly_Build_v3_MAC_THREE_TIER`.
- Restored the latest direct dependencies used by `app_streamlit.py`.
- Applied the Ollama internal-knowledge answer/source fix.
- Applied the latest ChatGPT-style question input flow (`chat_input_flow_v2`).
- Kept Enter and **질문** button submission.
- Removed Python caches, local indexes, generated strict evidence, and intermediate runtime artifacts.
- Kept `artifacts/week7/active_policy.json` because runtime/release code references it.
- Added GitHub-safe `.gitignore`, submission documents, evidence examples, and upload guidance.

## Functional evolution

- Core RAG baseline
- Semantic chunk experiments
- BGE-M3 / reranker candidate path
- Ollama / LangGraph / grounding fallback
- Evaluation and RAGAS
- LinUCB policy experiments with rollback
- Streamlit UI
- Strict staged release pipeline
- macOS MPS profile
- Production config / E2E hardening
- Safe document upload
- RAG -> Web -> Ollama internal-knowledge fallback

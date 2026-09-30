# IntellectAgent

**IntellectAgent** is a local-first AI question-answering project built around a gated RAG architecture.
It prefers verifiable evidence first and falls back in this order:

1. **RAG documents** — uploaded PDF/TXT/MD documents are searched first.
2. **Web search** — used only when the RAG corpus is empty.
3. **Ollama internal knowledge** — used only when no RAG documents exist and web evidence is unavailable; the UI marks this as **not externally verified**.

The project was developed as a 9-week final project and later hardened for macOS/Apple Silicon, production-style configuration, strict validation, safe document upload, and a ChatGPT-style Streamlit interface.

## Main features

- Safe PDF/TXT/MD upload and indexing from Streamlit
- BGE-M3 embeddings
- ChromaDB vector retrieval
- BGE reranking
- Local answer generation with Ollama (`qwen2.5:14b` by default)
- LangGraph workflow support
- Grounding / fallback behavior
- RAGAS and deterministic evaluation utilities
- Random / Rule / LinUCB policy comparison with rollback
- macOS Apple Silicon MPS runtime profile
- Smoke, Strict, Production E2E, and Release Gate scripts
- Three-tier answer path: **RAG → Web → Ollama knowledge**
- Chat history with Enter or **질문** button submission

## Answer flow

```text
User question
   |
   +-- Are RAG documents available?
   |      |
   |      +-- YES -> BGE-M3 -> Chroma -> BGE Reranker -> Ollama -> Grounding -> document sources
   |      |
   |      +-- NO  -> DDGS web search
   |                   |
   |                   +-- usable web evidence -> Ollama grounded answer -> web URLs
   |                   |
   |                   +-- web unavailable/no evidence -> Ollama internal knowledge
   |                                                   -> "external verification unavailable"
```

## Repository structure

```text
IntellectAgent/
├── app_streamlit.py          # Streamlit UI
├── src/intellectagent/       # Core package
├── config/                   # Smoke / production / release policies
├── scripts/
│   ├── mac/                  # macOS prepare, strict, promotion, start scripts
│   ├── strict/               # staged strict validation
│   └── week1..week9/         # weekly development gates
├── tests/                    # regression tests
├── data/
│   ├── sample_docs/          # public sample documents
│   ├── eval/                 # evaluation sets
│   └── real_docs/            # local user documents; ignored by Git
├── docs/                     # architecture, runbooks, final/submission docs
├── artifacts/week7/active_policy.json
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

## Requirements

- Python **3.11.x**
- Ollama
- macOS is the primary final-project environment
- Apple Silicon uses PyTorch MPS when available

> Install PyTorch for your platform first, then install this project's requirements.

## Quick start on macOS

```bash
cd IntellectAgent
python3.11 -m venv .venv-mac
source .venv-mac/bin/activate

python -m pip install --upgrade pip
# Install PyTorch appropriate for your Mac first.
python -m pip install -r requirements.txt

bash scripts/mac/start_intellectagent.sh
```

Then open:

```text
http://localhost:8501
```

The startup script checks the environment, configures `PYTHONPATH`, checks Ollama on port `11434`, and starts Streamlit.

## Manual Streamlit start

```bash
source .venv-mac/bin/activate
export PYTHONPATH="$PWD/src"
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export TF_CPP_MIN_LOG_LEVEL=3
export INTELLECTAGENT_CONFIG="$PWD/config/core.production.json"
export RUNTIME_PROFILE=mac-native
export OLLAMA_MODEL=qwen2.5:14b

python -m streamlit run app_streamlit.py
```

## Ollama check

```bash
lsof -iTCP:11434 -sTCP:LISTEN
curl -fsS http://127.0.0.1:11434/api/tags
ollama list
```

If port `11434` is free, run `ollama serve`. If Ollama is already listening and `/api/tags` responds, reuse it.

## Uploading RAG documents

Use the **문서 관리** area in the Streamlit sidebar.

1. Select PDF/TXT/MD files.
2. Click **문서 업로드 + 인덱싱**.
3. Files are validated and stored under `data/real_docs/`.
4. BGE-M3 embeddings are generated.
5. ChromaDB is updated.
6. Ask questions from the input box at the bottom of the chat.

`data/real_docs/` is intentionally ignored by Git so private documents are not accidentally uploaded to GitHub.

## Tests

Fast regression test:

```bash
PYTHONPATH=src python -m pytest -q
```

Smoke gate:

```bash
PYTHONPATH=src python scripts/run_release_gate.py --mode smoke
```

Production core strict gate:

```bash
PYTHONPATH=src python scripts/run_release_gate.py \
  --mode strict \
  --config config/core.production.json \
  --eval data/eval/week6/eval20.jsonl
```

Full Mac staged validation:

```bash
DATA_ROOT=data/real_docs \
EVAL_FILE=data/eval/week6/eval20.jsonl \
OLLAMA_MODEL=qwen2.5:14b \
bash scripts/mac/02_run_mac_strict.sh
```

Final promotion is intentionally fail-closed: if STEP14 or required production evidence does not pass, STEP15 refuses promotion.

## Development principles

- **Core before Agent** — the core retrieval path works independently of optional orchestration.
- **Evaluation before Expansion** — new features are measured before promotion.
- **Fail closed** — insufficient grounding results in fallback rather than fabricated certainty.
- **Optional isolation** — failures in optional components should not silently corrupt the baseline.
- **Evidence-based release** — tests, environment checks, and release gates are separated from ordinary UI success.

## Development history

The project progressed through nine weekly builds:

1. Environment, data inventory, safe loading, baseline RAG
2. Chunking strategies and evaluation set
3. Retrieval, reranking, candidate-k tuning
4. Ollama, LangGraph, grounding, fallback
5. Walking skeleton and P0 freeze
6. Evaluation, RAGAS, human evaluation
7. Random/Rule/LinUCB policy experiments and rollback
8. Streamlit UI, feedback, user testing
9. Final integration, reproduction, release gate

Post-project hardening added macOS MPS support, production configuration, production E2E checks, web document upload, web fallback, Ollama-knowledge fallback, and the chat-style UI.

See `docs/` for detailed runbooks and Korean submission documents.

## Privacy and repository hygiene

This repository intentionally does **not** commit:

- `data/real_docs/*`
- local Chroma / SQLite indexes
- generated evidence from local strict runs
- virtual environments
- model caches
- `.env` secrets

Only sample documents and selected example evidence under `docs/evidence_examples/` are included.

## Project status

This is an educational final-project prototype with a production-oriented validation structure. It demonstrates the architecture and verification workflow, but deployment-specific security, monitoring, model provenance, and operational controls should be strengthened before real production use.

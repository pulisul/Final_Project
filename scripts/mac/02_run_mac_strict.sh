#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
source "${VENV:-.venv-mac}/bin/activate"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export RUNTIME_PROFILE=mac-native
export INTELLECTAGENT_CONFIG="config/core.production.json"
MODEL="${OLLAMA_MODEL:-qwen2.5:14b}"
DATA_ROOT="${DATA_ROOT:-data/real_docs}"
EVAL_FILE="${EVAL_FILE:-data/eval/week6/eval20.jsonl}"
python scripts/strict/19_step6_9_gate.py --profile mac-native --root "$DATA_ROOT" --eval "$EVAL_FILE"
python scripts/strict/25_step10_13_gate.py --model "$MODEL" --full-ragas
python scripts/strict/27_step14_integrated_release_candidate.py --config "$INTELLECTAGENT_CONFIG" --profile mac-native --model "$MODEL" --root "$DATA_ROOT" --eval "$EVAL_FILE"

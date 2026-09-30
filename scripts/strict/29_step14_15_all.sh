#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
MODEL="${OLLAMA_MODEL:-qwen2.5:14b}"
DATA_ROOT="${DATA_ROOT:-data/real_docs}"
EVAL_FILE="${EVAL_FILE:-data/eval/week6/eval20.jsonl}"
PROFILE="${RUNTIME_PROFILE:-auto}"
python scripts/strict/27_step14_integrated_release_candidate.py --profile "$PROFILE" --model "$MODEL" --root "$DATA_ROOT" --eval "$EVAL_FILE"
python scripts/strict/28_step15_promote_final_release.py

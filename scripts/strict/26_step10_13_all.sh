#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
MODEL="${OLLAMA_MODEL:-qwen2.5:14b}"
python scripts/strict/25_step10_13_gate.py --model "$MODEL" "$@"

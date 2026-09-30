#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
MODEL="${OLLAMA_MODEL:-qwen2.5:14b}"
python scripts/strict/08_integrated_strict.py --model "$MODEL"
python scripts/strict/09_promote_release.py

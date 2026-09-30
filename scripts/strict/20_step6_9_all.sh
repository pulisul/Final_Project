#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"; export PYTHONPATH="$ROOT/src"
python scripts/strict/19_step6_9_gate.py "$@"

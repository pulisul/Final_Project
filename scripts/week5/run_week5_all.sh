#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# run_week5_gate already executes the regression suite, Walking Skeleton,
# P0 freeze creation/verification, and blocker check. Avoid duplicate work.
python scripts/week5/run_week5_gate.py

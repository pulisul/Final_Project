#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/week4/run_e2e_smoke.py
python -m pytest -q
python scripts/week4/run_week4_gate.py

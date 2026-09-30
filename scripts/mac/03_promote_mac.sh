#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
source "${VENV:-.venv-mac}/bin/activate"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/strict/28_step15_promote_final_release.py

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV="${VENV:-.venv-repro}"
"$PYTHON_BIN" -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel
# PyTorch may need a platform/CUDA-specific install before this step.
python -m pip install -r requirements-dev.txt
python -m pip freeze | sort > requirements-lock.txt
python -m pip check
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/week9/check_clean_repro.py
PYTHONPATH=src python scripts/week9/run_week9_gate.py --strict

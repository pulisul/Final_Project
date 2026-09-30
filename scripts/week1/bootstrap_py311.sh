#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
# PyTorch는 CPU/CUDA 환경에 맞는 공식 설치 명령으로 먼저 설치하세요.
python -m pip install -r requirements-dev.txt
python scripts/week1/check_environment.py || true
python -m pytest -q
python scripts/run_release_gate.py --mode smoke

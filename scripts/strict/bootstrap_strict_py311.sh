#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV="${VENV:-.venv-strict}"
"$PYTHON_BIN" -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel
# PyTorch is deliberately not forced here because CPU/CUDA wheels differ by host.
echo "[ACTION REQUIRED] Install the PyTorch build appropriate for this machine before continuing if not already present."
python - <<'PY'
try:
 import torch; print('torch',torch.__version__,'cuda',torch.cuda.is_available())
except Exception as e:
 print('torch_not_ready:',e)
PY
python -m pip install -r requirements-dev.txt
python -m pip freeze | LC_ALL=C sort > requirements-lock.txt
python -m pip check
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q
python scripts/strict/00_preflight.py
cat <<EOF
Strict environment prepared.
Next:
  1) ensure Ollama server is running and model exists: ${OLLAMA_MODEL:-qwen2.5:14b}
  2) run: bash scripts/strict/run_strict_all.sh
EOF

#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV="${VENV:-.venv-mac}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "[FAIL] python3.11 없음. Homebrew 예: brew install python@3.11"; exit 2; }
[[ "$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.11" ]] || { echo "[FAIL] Python 3.11 필요"; exit 2; }
if [[ "${RECREATE_VENV:-0}" == "1" ]]; then rm -rf "$VENV"; fi
[[ -d "$VENV" ]] || "$PYTHON_BIN" -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel
# macOS wheel은 CUDA wheel과 다르므로 일반 PyPI torch를 명시적으로 먼저 설치
python -m pip install torch
python - <<'PY'
import platform, torch
mps=bool(getattr(torch.backends,'mps',None) and torch.backends.mps.is_available())
print({'macOS':platform.mac_ver()[0],'arch':platform.machine(),'torch':torch.__version__,'mps_available':mps,'selected_device':'mps' if mps else 'cpu'})
PY
python -m pip install -r requirements-dev.txt
python -m pip freeze | LC_ALL=C sort > requirements-lock.txt
python -m pip check
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q
python scripts/strict/00_preflight.py
python scripts/strict/14_step6_gpu_runtime.py --profile mac-native
printf '\n[PASS] Mac 개발/Strict 기반환경 준비 완료\n'

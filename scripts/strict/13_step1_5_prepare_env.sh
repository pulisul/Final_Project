#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV="${VENV:-.venv-strict}"
EVIDENCE_DIR="$ROOT/evidence/strict"
mkdir -p "$EVIDENCE_DIR"
LOG="$EVIDENCE_DIR/STEP1_5_PREPARE.log"

exec > >(tee -a "$LOG") 2>&1

fail() {
  echo "[FAIL] $*" >&2
  exit 2
}

step() {
  echo
  echo "============================================================"
  echo "$*"
  echo "============================================================"
}

step "STEP 1 — Python 3.11 확인"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "$PYTHON_BIN 을 찾을 수 없습니다. Ubuntu에서는 python3.11 및 python3.11-venv 설치가 필요합니다."
PYVER="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "$PYVER" == "3.11" ]] || fail "Python 3.11이 필요하지만 현재 $PYVER 입니다."
"$PYTHON_BIN" --version

step "STEP 2 — 새 Strict 가상환경 생성"
if [[ -d "$VENV" ]]; then
  echo "[INFO] 기존 $VENV 발견"
  if [[ "${RECREATE_VENV:-0}" == "1" ]]; then
    echo "[INFO] RECREATE_VENV=1 → 기존 가상환경 삭제 후 재생성"
    rm -rf "$VENV"
  fi
fi
if [[ ! -d "$VENV" ]]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"
python --version
python -m pip install --upgrade pip setuptools wheel

step "STEP 3A — PyTorch 사전 확인"
set +e
python - <<'PY'
import json
try:
    import torch
    print(json.dumps({
        "installed": True,
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }, ensure_ascii=False, indent=2))
    raise SystemExit(0)
except Exception as e:
    print(json.dumps({"installed": False, "error": str(e)}, ensure_ascii=False, indent=2))
    raise SystemExit(3)
PY
TORCH_RC=$?
set -e

if [[ $TORCH_RC -ne 0 ]]; then
  cat <<'MSG'
[ACTION REQUIRED] PyTorch가 아직 설치되지 않았습니다.
macOS라면 활성 venv에서 `python -m pip install torch`로 macOS용 PyTorch wheel을 설치하십시오.
NVIDIA GPU 서버라면 해당 Driver/CUDA 조합에 맞는 공식 PyTorch wheel을 설치하십시오.
CPU 서버라면 CPU용 PyTorch를 명시적으로 설치하십시오.
설치 후 이 스크립트를 다시 실행하십시오.

이 스크립트는 잘못된 CPU/GPU PyTorch를 자동 선택하지 않기 위해 여기서 중단합니다.
MSG
  exit 3
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[INFO] NVIDIA GPU 감지"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  CUDA_OK="$(python - <<'PY'
import torch
print('1' if torch.cuda.is_available() else '0')
PY
)"
  [[ "$CUDA_OK" == "1" ]] || fail "NVIDIA GPU가 보이지만 현재 PyTorch에서 CUDA를 사용할 수 없습니다. CPU용 torch가 설치되었거나 CUDA 호환 문제가 있을 수 있습니다."
elif [[ "$(uname -s)" == "Darwin" ]]; then
  echo "[INFO] macOS 감지 — NVIDIA/CUDA는 요구하지 않습니다."
  python - <<'PY'
import platform, torch
mps=bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
print({"arch":platform.machine(),"mps_available":mps,"selected_device":"mps" if mps else "cpu"})
PY
else
  echo "[WARN] nvidia-smi가 없습니다. CPU 환경이라면 정상입니다. GPU 사용 예정이라면 Driver 상태를 먼저 확인하십시오."
fi

step "STEP 3B — 프로젝트 의존성 설치"
python -m pip install -r requirements-dev.txt

step "STEP 4 — 의존성 잠금 및 pip check"
python -m pip freeze | LC_ALL=C sort > requirements-lock.txt
[[ -s requirements-lock.txt ]] || fail "requirements-lock.txt 생성 실패"
python -m pip check

step "STEP 5A — 전체 회귀테스트"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q

step "STEP 5B — Strict Preflight"
python scripts/strict/00_preflight.py

step "STEP 1~5 완료"
echo "[PASS] Python 3.11 / venv / dependencies / lock / pip check / pytest / preflight"
echo "Evidence: $EVIDENCE_DIR/00_preflight.json"
echo "Log: $LOG"
echo "Next: python scripts/strict/10_diagnose_runtime.py --model ${OLLAMA_MODEL:-qwen2.5:14b}"

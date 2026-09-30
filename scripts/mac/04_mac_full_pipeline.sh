#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export RUNTIME_PROFILE=mac-native
export INTELLECTAGENT_CONFIG="config/core.production.json"
MODEL="${OLLAMA_MODEL:-qwen2.5:14b}"
DATA_ROOT="${DATA_ROOT:-data/real_docs}"
EVAL_FILE="${EVAL_FILE:-data/eval/week6/eval20.jsonl}"

bash scripts/mac/00_check_mac.sh
bash scripts/mac/01_prepare_mac.sh

if ! command -v ollama >/dev/null 2>&1; then
  echo "[FAIL] Ollama가 없습니다. Homebrew 예: brew install ollama"; exit 2
fi
if ! python - <<'PY'
import socket
try:
    with socket.create_connection(('127.0.0.1',11434),timeout=0.5): pass
except OSError: raise SystemExit(2)
PY
then
  echo "[FAIL] Ollama 서버가 127.0.0.1:11434에서 실행 중이지 않습니다. 다른 터미널에서 'ollama serve'를 실행하십시오."; exit 2
fi
if ! ollama list | awk 'NR>1 {print $1}' | grep -Fxq "$MODEL"; then
  echo "[FAIL] Ollama 모델 '$MODEL'이 없습니다. 예: ollama pull $MODEL"; exit 2
fi
if [ ! -d "$DATA_ROOT" ] || [ -z "$(find "$DATA_ROOT" -type f -print -quit 2>/dev/null)" ]; then
  echo "[FAIL] 실제 데이터가 없습니다: $DATA_ROOT"; exit 2
fi

DATA_ROOT="$DATA_ROOT" EVAL_FILE="$EVAL_FILE" OLLAMA_MODEL="$MODEL" bash scripts/mac/02_run_mac_strict.sh
bash scripts/mac/03_promote_mac.sh

echo "[PASS] Mac 전체 production 파이프라인 완료"
echo "Config: $INTELLECTAGENT_CONFIG"
echo "Promotion: artifacts/strict/FINAL_RELEASE_PROMOTION.json"

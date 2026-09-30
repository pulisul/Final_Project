#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
VENV="${VENV:-.venv-mac}"

if [[ ! -d "$VENV" ]]; then
  echo "[FAIL] $VENV 가상환경이 없습니다. 먼저 실행하세요:"
  echo "  bash scripts/mac/01_prepare_mac.sh"
  exit 2
fi

# shellcheck disable=SC1090
source "$VENV/bin/activate"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export TF_CPP_MIN_LOG_LEVEL=3
export INTELLECTAGENT_CONFIG="${INTELLECTAGENT_CONFIG:-$ROOT/config/core.production.json}"
export RUNTIME_PROFILE="${RUNTIME_PROFILE:-mac-native}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:14b}"

echo "=== Ollama 11434 포트 확인 ==="
if ! lsof -iTCP:11434 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "11434 비어 있음 → ollama serve 실행"
  nohup ollama serve > /tmp/ollama-intellectagent.log 2>&1 &
  sleep 3
else
  if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "11434에서 Ollama가 정상 LISTEN → 그대로 사용"
  else
    echo "11434에 꼬인 Ollama 프로세스 존재 → 종료 후 ollama serve"
    pkill -f ollama || true
    sleep 2
    nohup ollama serve > /tmp/ollama-intellectagent.log 2>&1 &
    sleep 3
  fi
fi

if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "[FAIL] Ollama API가 응답하지 않습니다."
  echo "로그: tail -100 /tmp/ollama-intellectagent.log"
  exit 2
fi

echo "[PASS] Ollama API"

if ! ollama list | awk 'NR>1 {print $1}' | grep -Fxq "$OLLAMA_MODEL"; then
  echo "[WARN] $OLLAMA_MODEL 모델이 없습니다."
  echo "필요하면 실행: ollama pull $OLLAMA_MODEL"
fi

if ! python -c 'import ddgs' >/dev/null 2>&1; then
  echo "[FAIL] 웹 fallback용 ddgs 패키지가 없습니다."
  echo "다음을 한 번 실행하세요: python -m pip install -r requirements.txt"
  exit 2
fi

echo "=== IntellectAgent Streamlit 시작 ==="
exec python -m streamlit run app_streamlit.py

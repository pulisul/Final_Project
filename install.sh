#!/usr/bin/env bash
# IntellectAgent 설치 (macOS / Linux).  압축을 푼 폴더에서:  bash install.sh [옵션]
# 옵션은 scripts/setup_env.py 와 같다:  --yes  --dry-run  --skip-model  --skip-embedding  --skip-tests  --recreate  --install-ollama  --model NAME
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

is311() { "$1" -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 1)' >/dev/null 2>&1; }
PY=""
for c in "${PYTHON_BIN:-}" python3.11 /opt/homebrew/bin/python3.11 /usr/local/bin/python3.11 "$HOME/.pyenv/shims/python3.11" python3 python; do
  [[ -n "$c" ]] || continue
  if command -v "$c" >/dev/null 2>&1 && is311 "$(command -v "$c")"; then PY="$(command -v "$c")"; break; fi
done
if [[ -z "$PY" ]]; then
  echo "[FAIL] Python 3.11 을 찾지 못했습니다 (이 프로젝트는 3.11 만 지원합니다)."
  case "$(uname -s)" in
    Darwin) echo "  설치:  brew install python@3.11     (Homebrew 가 없으면 https://brew.sh 먼저)";;
    Linux)  echo "  설치:  sudo apt install python3.11 python3.11-venv     (우분투/데비안)  또는 pyenv install 3.11";;
  esac
  echo "  설치한 뒤 다시:  bash install.sh   (경로를 직접 줄 때: PYTHON_BIN=/경로/python3.11 bash install.sh)"
  exit 2
fi
echo "Python: $PY ($("$PY" --version 2>&1))"
exec "$PY" scripts/setup_env.py "$@"

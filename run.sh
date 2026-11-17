#!/usr/bin/env bash
# IntellectAgent 실행 (macOS / Linux).  사용법:  bash run.sh <명령>   (명령 목록: bash run.sh --help)
#   doctor | app | test | redteam | docs [폴더] | qa [폴더] | eval [폴더] | setup-model
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=""
for c in .venv-real/bin/python python3.11 python3 python; do
  if [[ -x "$c" ]] || command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
[[ -n "$PY" ]] || { echo "[FAIL] 파이썬이 없습니다. 먼저 bash install.sh"; exit 2; }
exec "$PY" manage.py "$@"

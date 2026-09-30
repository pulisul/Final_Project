#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
[[ "$(uname -s)" == "Darwin" ]] || { echo "[FAIL] 이 스크립트는 macOS 전용입니다."; exit 2; }
echo "macOS: $(sw_vers -productVersion 2>/dev/null || true)"
echo "Arch : $(uname -m)"
echo "CPU  : $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
MEM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
python3 - <<PY
b=int('$MEM_BYTES' or 0)
print('Memory GB:', round(b/(1024**3),2) if b else 'unknown')
PY
for c in brew python3.11 ollama; do
  if command -v "$c" >/dev/null 2>&1; then echo "[OK] $c -> $(command -v "$c")"; else echo "[WARN] $c 없음"; fi
done
if command -v python3.11 >/dev/null 2>&1; then python3.11 --version; fi

# Strict Gate 현재 상태

현재 Smoke 최종 Gate는 통과했지만 Strict Gate는 **의도적으로 미통과** 상태입니다.

## 현재 차단 사유
1. 실행 환경 Python이 3.13.5이며 프로젝트 요구사항은 Python 3.11.x입니다.
2. 깨끗한 Python 3.11 환경에서 생성해야 할 `requirements-lock.txt`가 아직 없습니다.
3. 현재 샌드박스의 `pip check`에는 프로젝트와 무관한 전역 `moviepy`/`pillow` 충돌이 있습니다.
4. 현재 샌드박스에는 실제 Strict에 필요한 `ollama`, `langgraph`, `ragas`, `streamlit`이 설치되어 있지 않습니다.

## 실제 프로젝트 PC에서 해소하는 방법
```bash
PYTHON_BIN=python3.11 bash scripts/week9/lock_and_reproduce_py311.sh
```

이 명령은 새 가상환경 생성 → requirements 설치 → lock 생성 → `pip check` → 전체 pytest → 재현 검사 → Strict Gate 순으로 실행합니다.

Strict PASS 전까지 `WEEK9_GATE.json`의 Smoke PASS를 최종 성능/재현성 PASS로 오인하면 안 됩니다.

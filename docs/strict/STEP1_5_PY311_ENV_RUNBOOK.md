# Strict STEP 1~5 — Python 3.11 환경 준비 Runbook

목표는 AI 모델을 실행하기 전에 다음 다섯 조건을 확정하는 것입니다.

1. Python 3.11.x
2. 새 `.venv-strict`
3. 머신에 맞는 PyTorch 명시 설치
4. `requirements-lock.txt` + `pip check PASS`
5. 전체 pytest + Strict Preflight PASS

## 0. Ubuntu 사전 확인

```bash
cd IntellectAgent_Weekly_Build_v3
python3.11 --version
```

`Python 3.11.x`가 아니면 다음 단계로 가지 않습니다.

Ubuntu 저장소에 Python 3.11이 제공되는 환경이라면 예시는 다음과 같습니다.

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

> 배포판/사내 이미지 정책에 따라 패키지명이 다를 수 있으므로, 위 명령이 실패하면 OS 정책에 맞는 Python 3.11 설치 방법을 사용합니다.

## 1. PyTorch를 먼저 결정

이 프로젝트는 PyTorch wheel을 자동 선택하지 않습니다. GPU 서버에서 CPU용 torch가 먼저 깔리는 실수를 막기 위해서입니다.

먼저 GPU 확인:

```bash
nvidia-smi
```

그 다음 **해당 Driver/GPU와 호환되는 공식 PyTorch 설치 명령**으로 torch를 `.venv-strict` 안에 설치한 후 STEP 1~5 스크립트를 다시 실행합니다.

CPU 서버라면 CPU용 PyTorch를 명시적으로 설치합니다.

## 2. STEP 1~5 자동 실행

```bash
PYTHON_BIN=python3.11 \
VENV=.venv-strict \
bash scripts/strict/13_step1_5_prepare_env.sh
```

기존 가상환경을 완전히 버리고 다시 만들려면:

```bash
RECREATE_VENV=1 \
PYTHON_BIN=python3.11 \
VENV=.venv-strict \
bash scripts/strict/13_step1_5_prepare_env.sh
```

## 3. 첫 실행에서 PyTorch 미설치로 멈추는 경우

정상 동작입니다. 메시지는 대략 다음과 같습니다.

```text
[ACTION REQUIRED] PyTorch가 아직 설치되지 않았습니다.
```

이때 `.venv-strict`를 활성화한 후 환경에 맞는 torch를 설치합니다.

```bash
source .venv-strict/bin/activate
# 여기서 실제 서버에 맞는 공식 PyTorch 설치 명령 실행
```

설치 확인:

```bash
python - <<'PY'
import torch
print('torch=', torch.__version__)
print('cuda_available=', torch.cuda.is_available())
print('torch_cuda=', torch.version.cuda)
if torch.cuda.is_available():
    print('gpu=', torch.cuda.get_device_name(0))
PY
```

GPU 서버인데 `cuda_available=False`이면 다음 단계로 가지 않습니다.

## 4. requirements 설치 실패 시

먼저 현재 가상환경을 확인합니다.

```bash
which python
python --version
python -m pip --version
```

반드시 `.venv-strict` 아래 경로와 Python 3.11이어야 합니다.

그 뒤 개별 오류를 확인합니다.

```bash
python -m pip install -r requirements-dev.txt -v
```

패키지 resolver 충돌이 생기면 기존 시스템 환경을 고치는 대신 `.venv-strict`를 삭제하고 새로 만드는 편이 안전합니다.

## 5. pip check 실패 시

```bash
python -m pip check
```

출력된 충돌을 그대로 기록합니다. 관련 없는 시스템 패키지를 억지로 업/다운그레이드하지 말고, 프로젝트 requirements 범위 안에서 해결합니다.

수정 후 반드시:

```bash
python -m pip freeze | LC_ALL=C sort > requirements-lock.txt
python -m pip check
```

를 다시 실행합니다.

## 6. pytest 실패 시

```bash
PYTHONPATH=src python -m pytest -q
```

첫 실패만 자세히 볼 때:

```bash
PYTHONPATH=src python -m pytest -x -vv
```

Core 회귀가 발생하면 Strict 모델 검증으로 넘어가지 않습니다.

## 7. Preflight PASS 확인

```bash
PYTHONPATH=src python scripts/strict/00_preflight.py
```

최종 JSON:

```text
evidence/strict/00_preflight.json
```

최소 조건:

```json
{
  "python_311": true,
  "requirements_lock_exists": true,
  "pip_check": {"pass": true},
  "pass": true
}
```

여기까지가 STEP 1~5 완료입니다.

## 8. 다음 단계

Preflight PASS 후에만 다음을 실행합니다.

```bash
python scripts/strict/10_diagnose_runtime.py --model qwen2.5:14b
```

그 후 BGE-M3 → ChromaDB → BGE reranker → Ollama 순으로 승격합니다.

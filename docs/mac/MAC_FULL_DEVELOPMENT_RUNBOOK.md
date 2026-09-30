# IntellectAgent — Mac 전체 개발·Strict·Release Runbook

## 목표
Mac 한 대에서 개발 1~9주, BGE-M3, ChromaDB, BGE reranker, Ollama, LangGraph, RAGAS, Streamlit, STEP 14 통합 Strict와 STEP 15 PROMOTED까지 수행한다.

## 정식 Mac 프로필
- Apple Silicon: `mac-native`, accelerator=`mps` (가능하면 사용)
- Intel Mac: `mac-native`, accelerator=`cpu`
- CUDA는 요구하지 않는다.
- `PYTORCH_ENABLE_MPS_FALLBACK=1`을 켜 MPS 미지원 연산은 CPU fallback을 허용한다.
- 성능 수치는 Linux/CUDA 결과와 섞지 않고 Release evidence의 runtime_profile로 구분한다.

## 1. 도구 설치
Homebrew가 있다면 Python 3.11과 Ollama를 설치한다.

```bash
brew install python@3.11
brew install ollama
```

## 2. 프로젝트 환경
```bash
bash scripts/mac/01_prepare_mac.sh
```

이 단계가 `.venv-mac`, macOS용 PyTorch, requirements-lock, pip check, pytest, Preflight, mac-native runtime Gate를 만든다.

## 3. Ollama
```bash
ollama serve
```
다른 터미널에서:
```bash
ollama pull qwen2.5:14b
ollama list
```

메모리가 부족하면 `OLLAMA_MODEL`을 실제 설치한 더 작은 모델로 바꿀 수 있지만, 그 모델명을 Release evidence에 그대로 남겨야 한다.

## 4. 실제 데이터
최종 Promotion은 sample_docs를 거부한다. `data/real_docs`와 실제 `eval20.jsonl`의 relevant_doc_ids를 일치시킨다.

## 5. Mac 전체 Strict
```bash
DATA_ROOT=data/real_docs \
EVAL_FILE=data/eval/week6/eval20.jsonl \
OLLAMA_MODEL=qwen2.5:14b \
bash scripts/mac/02_run_mac_strict.sh
```

## 6. 최종 Promotion
STEP14가 PASS한 뒤에만:
```bash
bash scripts/mac/03_promote_mac.sh
```

성공 결과:
- `artifacts/strict/FINAL_RELEASE_PROMOTION.json`
- `evidence/strict/STEP15_PROMOTION_VERIFY.json`

## 성능 해석
Mac MPS/CPU latency와 메모리 수치는 NVIDIA/CUDA 결과와 직접 비교하지 않는다. 기능/정확성 Gate는 동일하게 유지하되 플랫폼 성능은 `runtime_profile`, `accelerator`, `architecture`와 함께 보고한다.

## 한 번에 실행하는 방법
사전 준비가 끝났고 `ollama serve`가 실행 중이며 모델을 내려받았다면 다음 한 명령으로 환경 구성부터 Promotion까지 연결할 수 있다.

```bash
DATA_ROOT=data/real_docs \
EVAL_FILE=data/eval/week6/eval20.jsonl \
OLLAMA_MODEL=qwen2.5:14b \
bash scripts/mac/04_mac_full_pipeline.sh
```

### Mac 메모리가 작을 때
`qwen2.5:14b`가 Mac의 통합 메모리에 부담이 되면 더 작은 Ollama 모델을 사용해도 코드 경로는 동일하다. 단, 최종 Release evidence에는 실제 사용한 모델명이 기록되므로 모델 변경 후 STEP14/15를 처음부터 다시 실행해야 한다.

### Apple Silicon과 Intel Mac
- Apple Silicon: PyTorch가 MPS를 사용할 수 있으면 `mps`가 자동 선택된다.
- Intel Mac: `cpu`가 자동 선택된다.
- 두 경우 모두 `mac-native`는 정식 Release 프로필이다.
- 성능 지표는 각 Mac 하드웨어에 귀속되며 CUDA 서버 수치와 동일 기준의 속도 비교로 해석하지 않는다.

# IntellectAgent Strict 실행 가이드

이 단계는 9주 개발 일정 이후의 **최종 검증 단계**이며 새 기능 개발이 아니다.

## 목표
Smoke 기준선을 보존한 채 실제 Python 3.11 환경에서 다음을 순서대로 승격한다.

1. Python 3.11 + requirements-lock + pip check
2. BGE-M3 embedding
3. ChromaDB persistence
4. BGE reranker
5. Ollama local LLM + grounding
6. LangGraph invoke
7. RAGAS local-Ollama 2-case smoke
8. Streamlit compile
9. Core strict release gate
10. Release promotion

## 준비
- Python 3.11.x
- Ollama 서버
- `qwen2.5:14b` 또는 `OLLAMA_MODEL`로 지정한 모델
- CPU/CUDA 환경에 맞는 PyTorch
- 디스크 여유 공간(BGE-M3, reranker, LLM 모델)

## 1. 새 환경
```bash
PYTHON_BIN=python3.11 bash scripts/strict/bootstrap_strict_py311.sh
```
PyTorch 설치 방식은 머신마다 다르므로 스크립트가 임의의 CUDA wheel을 강제하지 않는다.

## 2. Ollama 확인
```bash
ollama list
curl http://127.0.0.1:11434/api/tags
```
필요한 모델이 없다면 실제 프로젝트 PC에서 설치한다.

## 3. 단계별 실행
```bash
python scripts/strict/00_preflight.py
python scripts/strict/01_validate_bge_embedding.py
python scripts/strict/02_validate_chroma.py
python scripts/strict/03_validate_bge_reranker.py
python scripts/strict/04_validate_ollama.py --model qwen2.5:14b
python scripts/strict/05_validate_langgraph.py
python scripts/strict/06_validate_ragas.py --model qwen2.5:14b
python scripts/strict/07_validate_streamlit.py
python scripts/run_release_gate.py --mode strict
```

## 4. 통합 실행
```bash
OLLAMA_MODEL=qwen2.5:14b bash scripts/strict/run_strict_all.sh
```
통합 실행기는 첫 실패 지점 이후를 `blocked_by`로 표시한다. 앞 단계가 실패한 상태에서 뒤 단계 실패를 연쇄 오류로 오해하지 않게 하기 위함이다.

## 5. 승격 원칙
`evidence/strict/STRICT_INTEGRATED.json`의 `overall_pass=true`일 때만 `09_promote_release.py`가 `artifacts/strict/RELEASE_PROMOTION.json`을 만든다. FAIL 상태에서 릴리스 승격은 거부된다.

## 실패 대응
- preflight FAIL: Python/lock/pip 의존성부터 수정. AI 코드는 수정하지 않는다.
- BGE FAIL: 모델/torch/메모리 문제를 해결하고 Core smoke 기준선은 유지한다.
- Chroma FAIL: 저장 디렉터리/패키지 문제만 수정한다.
- Reranker FAIL: 기본 vector search를 유지한다.
- Ollama FAIL: 모델/server/grounding 프롬프트 문제를 분리한다.
- LangGraph FAIL: Core workflow는 유지하고 LangGraph 승격만 보류한다.
- RAGAS FAIL: 자체 deterministic 평가를 유지하고 RAGAS 승격만 보류한다.
- Streamlit FAIL: Core CLI/tests는 유지하고 UI 승격만 보류한다.

## 최종 증거
`evidence/strict/`에 단계별 JSON을 저장하며 최종 성공 후 `RELEASE_PROMOTION.json`을 보관한다.

## 장애 진단 및 실패 지점부터 재개

Strict 실행 전에 런타임을 진단합니다.

```bash
python scripts/strict/10_diagnose_runtime.py --model qwen2.5:14b
```

주요 검사 항목은 디스크 여유 공간, NVIDIA GPU/드라이버 인식, PyTorch/CUDA 인식, Ollama 11434 포트, Ollama 모델 설치 여부입니다.

중간 단계까지 PASS한 뒤 실패했다면 처음부터 모두 재실행하지 않고 다음 명령으로 재개할 수 있습니다.

```bash
OLLAMA_MODEL=qwen2.5:14b bash scripts/strict/12_resume_strict.sh
```

`--resume`은 이전 `STRICT_INTEGRATED.json`에서 PASS한 단계만 재사용하고 첫 미통과 단계부터 다시 실행합니다. 환경이나 모델 파일을 변경한 경우에는 재사용하지 말고 일반 `run_strict_all.sh`로 전체 재검증하는 것이 안전합니다.

최종 원인 요약은 다음 파일에 저장됩니다.

```text
evidence/strict/STRICT_SUMMARY.json
```

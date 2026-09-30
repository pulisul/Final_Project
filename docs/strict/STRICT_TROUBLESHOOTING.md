# Strict 실패 진단표

| 단계 | 대표 실패 | 먼저 확인할 것 | Core 처리 |
|---|---|---|---|
| 00 | Python 3.11 아님 / lock 없음 | `python -V`, `pip check` | Core 수정 금지 |
| 01 | BGE-M3 load/OOM | torch build, RAM/VRAM, HF cache | Hash smoke 유지 |
| 02 | Chroma import/write 실패 | 패키지, 디렉터리 권한 | InMemory smoke 유지 |
| 03 | reranker OOM/점수 이상 | CrossEncoder, candidate size | 기본검색 유지 |
| 04 | Ollama 연결 실패 | 11434, `ollama list`, 모델명 | Extractive smoke 유지 |
| 05 | LangGraph import/invoke 실패 | 설치버전, state contract | Core workflow 유지 |
| 06 | RAGAS API/structured output 실패 | local endpoint, 2-case smoke | 자체 평가 유지 |
| 07 | Streamlit 미설치/compile 실패 | package, app syntax | Core 정상 |
| 08 | Core strict FAIL | lock, thresholds, full test | 릴리스 금지 |

절대 원칙: FAIL을 없애기 위해 threshold를 낮추거나 증거 JSON을 수동으로 PASS로 바꾸지 않는다.

## 추가 자동 진단

`10_diagnose_runtime.py`는 다음을 구분합니다.

- `disk_space`: 모델/인덱스 설치 전에 디스크 여유 공간 부족
- `torch_missing` / `torch_broken`: PyTorch 미설치 또는 import 실패
- `nvidia_detected_but_torch_cuda_unavailable`: GPU는 보이지만 PyTorch가 CUDA를 사용하지 못함
- `ollama_port_closed`: Ollama 서버가 127.0.0.1:11434에서 응답하지 않음
- `ollama_model_missing_or_ollama_unavailable`: 지정 모델이 없거나 Ollama 명령 자체가 없음

`11_build_strict_report.py`는 Strict 실패 로그에서 `missing_python_dependency`, `cuda_or_gpu`, `service_unreachable`, `disk_space`, `environment_preflight`, `timeout` 등을 분류하고 다음 조치(`next_action`)를 기록합니다.

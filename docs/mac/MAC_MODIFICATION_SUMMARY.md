# Mac 전체 실행 전환 요약

## 변경 목적
기존 NVIDIA/CUDA 중심 Strict 경로를 플랫폼 프로필 기반으로 변경하여 macOS에서도 개발 1~9주, 실제 BGE-M3/reranker, ChromaDB, Ollama, LangGraph, RAGAS, Streamlit, STEP14 통합 Strict, STEP15 Release Promotion까지 수행할 수 있게 했다.

## 주요 변경
1. `platform_runtime.py`: `auto`, `mac-native`, `linux-cuda`, `cpu` 프로필과 장치 자동 선택.
2. Apple Silicon은 MPS가 있으면 `mps`, 없으면 CPU. Intel Mac은 CPU.
3. BGE-M3와 CrossEncoder에 명시적 `device` 주입.
4. MPS 메모리와 macOS 통합 메모리 증거 기록.
5. STEP6~9 Gate가 CUDA를 강제하지 않고 프로필별 Runtime Gate 사용.
6. STEP14/15에 runtime profile을 기록하고 Mac 정식 Promotion 허용.
7. `scripts/mac/`에 Mac 전용 사전점검, 환경구성, Strict, Promotion, 전체 실행기 추가.
8. `PYTORCH_ENABLE_MPS_FALLBACK=1`로 MPS 미지원 연산의 CPU fallback 지원.

## 최종 실행
```bash
bash scripts/mac/00_check_mac.sh
bash scripts/mac/01_prepare_mac.sh
# 별도 터미널: ollama serve
# 모델: ollama pull qwen2.5:14b
DATA_ROOT=data/real_docs EVAL_FILE=data/eval/week6/eval20.jsonl OLLAMA_MODEL=qwen2.5:14b bash scripts/mac/04_mac_full_pipeline.sh
```

## 완료 판정
`artifacts/strict/FINAL_RELEASE_PROMOTION.json`이 생성되고 `release_status`가 `PROMOTED`이며 `evidence/strict/STEP15_PROMOTION_VERIFY.json`의 `pass`가 true여야 최종 완료다.

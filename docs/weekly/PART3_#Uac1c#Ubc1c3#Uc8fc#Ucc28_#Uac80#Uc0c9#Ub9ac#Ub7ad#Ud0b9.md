# PART 3 — 개발 3주차: 검색·리랭킹·지연시간·복구 Gate

## 주간 목표
이번 주에는 청킹 전략을 바꾸지 않고 검색 계층만 개선한다. 실험 변수는 `candidate_k`, `top_k`, reranker뿐이다.

## 준비 라이브러리
- 필수 smoke: Python 표준 라이브러리, pytest
- 실제 BGE: `sentence-transformers`, PyTorch
- Vector DB 실환경 검증 시: `chromadb`

## 고정값
- seed: 20260831
- top_k: 5
- candidate_k 후보: 5, 10, 20 중 최대 3개만 비교
- production reranker 후보: `BAAI/bge-reranker-v2-m3`
- 튜닝 종료 규칙: 위 범위를 벗어난 추가 탐색 금지

## Day 1 — baseline top-k 검색
1. Week 2에서 승격된 청킹 전략을 고정한다.
2. reranker 없이 `candidate_k=10, top_k=5`를 실행한다.
3. Hit Rate@k, MRR, 평균/P95 latency를 저장한다.

```bash
PYTHONPATH=src python scripts/week3/run_retrieval_benchmark.py \
  --reranker none --candidate-k 10 --top-k 5 \
  --out evidence/week3/none_c10_k5.json
```

## Day 2 — 리랭커 계약 및 실패 복구
`safe_rerank()`는 reranker 예외를 잡고 원래 vector search 순서를 그대로 반환한다. 리랭커는 품질 향상 기능이지 Core RAG의 단일 장애점이 아니다.

## Day 3 — 결정적 smoke reranker
실제 BGE 설치 없이 테스트하기 위해 `TokenOverlapReranker`를 사용한다. 이 결과는 production 품질 증거가 아니다.

```bash
PYTHONPATH=src python scripts/week3/run_retrieval_benchmark.py \
  --reranker token-overlap --candidate-k 10 --top-k 5 \
  --out evidence/week3/token-overlap_c10_k5.json
```

## Day 4 — 실제 BGE smoke
Python 3.11 실환경에서 실행한다.

```bash
python scripts/week3/validate_bge_reranker.py --model BAAI/bge-reranker-v2-m3
```

PASS 조건:
- score가 모두 finite
- 관련 문장 score > 무관 문장 score
- 결과 JSON 보관

이후 실제 benchmark:

```bash
PYTHONPATH=src python scripts/week3/run_retrieval_benchmark.py \
  --reranker bge --model BAAI/bge-reranker-v2-m3 \
  --candidate-k 10 --top-k 5 \
  --out evidence/week3/bge_c10_k5.json
```

## Day 5 — candidate_k 튜닝
후보는 5/10/20만 사용한다. 품질 개선이 없는데 latency만 늘어나면 작은 값을 선택한다. 추가 후보를 즉흥적으로 늘리지 않는다.

## Day 6 — 비교·문서화·Gate
선택 우선순위는 사전에 고정한다.
1. Hit Rate@k
2. MRR
3. 품질 동률이면 latency가 baseline의 1.5배 이하인지

Smoke 전체 실행:

```bash
bash scripts/week3/run_week3_all.sh
```

실제 BGE 승격 Gate:

```bash
python scripts/week3/run_week3_gate.py --strict-bge
```

## Week 3 완료 Gate
- 전체 pytest 실패 0
- baseline 결과 존재
- rerank 결과 존재
- Hit Rate@k >= 0.70
- MRR >= 0.60
- reranker 예외 시 기본 검색 복구 테스트 PASS
- strict 승격 시 실제 BGE smoke PASS
- strict 승격 시 실제 BGE benchmark 결과 존재
- 튜닝 범위 5/10/20 이외로 확대하지 않음

## 힘든 지점과 극복
- CrossEncoder가 느림: candidate_k를 20→10→5 순으로 줄여 품질/속도 균형을 확인한다.
- BGE 모델 설치 문제: Core 테스트는 deterministic reranker로 유지하고 BGE 승격만 보류한다.
- 점수가 기대 이하: 리랭커를 억지로 승격하지 않고 baseline을 유지한다.

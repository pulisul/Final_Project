# PART 2 — 개발 2주차: 청킹 비교 실험과 재현성 Gate

## 이번 주 목표
이번 주의 목적은 "의미 단위 청킹이 더 좋을 것"이라는 가정을 증명하는 것이 아니라, **동일 문서·동일 평가질문·동일 임베더·동일 top-k·동일 seed에서 청킹 전략만 바꾸어 공정하게 비교**하는 것입니다.

## 준비 라이브러리
필수: Python 3.11, pytest, numpy. 실제 임베딩 평가 시 sentence-transformers + torch + BAAI/bge-m3. Smoke에서는 HashEmbedder를 사용하므로 GPU가 없어도 실험 코드 자체를 검증할 수 있습니다.

## Day 1 — 평가질문 15개 확정
1. `python scripts/week2/build_eval15.py`
2. 생성물: `data/eval/week2/eval15.jsonl`
3. 샘플용 질문은 코드 동작 검증용입니다. 실제 제출용에서는 실문서 기반 15개로 반드시 교체합니다.
4. 각 질문은 relevant_doc_ids, should_fallback을 포함합니다.

완료 기준: 정확한 근거 문서를 사람이 확인한 15개 질문이 존재하고, 그중 일부는 multi_doc/fallback 유형이어야 합니다.

## Day 2 — 고정 길이 청킹 전략 구현
`src/intellectagent/chunk_strategies.py`에서 `semantic`과 `length`를 동일 인터페이스로 제공합니다. 길이 전략은 size=700, overlap=80을 기본으로 하고 chunk ID는 입력과 설정에 대해 결정적입니다.

테스트: `pytest tests/week2/test_chunk_strategies.py -q`

## Day 3 — 실험 설정 저장 + 시드 고정
실험마다 `RetrievalExperimentConfig`를 JSON으로 저장합니다. 기본 seed는 20260824입니다. seed, top_k, chunk size/overlap이 기록되지 않은 실험은 최종 근거로 사용하지 않습니다.

## Day 4 — 의미 청킹 실험
`python scripts/week2/run_chunk_experiment.py --strategy semantic`

생성물:
- `evidence/week2/semantic_config.json`
- `evidence/week2/semantic_result.json`

## Day 5 — 길이 청킹 실험
`python scripts/week2/run_chunk_experiment.py --strategy length`

생성물:
- `evidence/week2/length_config.json`
- `evidence/week2/length_result.json`

## Day 6 — 비교·결정
`python scripts/week2/compare_chunk_strategies.py`

우선순위:
1. Hit Rate@k
2. MRR
3. 둘이 같을 때 평균 latency

예상과 다른 결과가 나와도 바꾸지 않습니다. "가설 검증"이 아니라 "사실 확인"이 목적입니다.

## 주간 일괄 실행
```bash
PYTHONPATH=src bash scripts/week2/run_week2_all.sh
```

## Week 2 Gate
다음이 모두 통과해야 개발 3주차로 넘어갑니다.
- 전체 pytest 실패 0
- 평가질문 15개 존재
- semantic/length 설정 JSON 존재
- semantic/length 결과 JSON 존재
- 비교 결과 존재
- 선택된 전략의 Hit Rate@k >= 0.70
- 선택된 전략의 MRR >= 0.60

샘플 문서 결과는 smoke evidence일 뿐이며 실제 프로젝트 승격은 실문서 평가셋으로 다시 통과해야 합니다.

## 실패 시 대응
- 평가질문이 애매함 → 질문/정답 문서 관계부터 수정. 청킹을 먼저 튜닝하지 않음.
- 양쪽 지표 모두 낮음 → 임베더/문서 품질/평가셋을 진단. 청킹값 무한 튜닝 금지.
- semantic과 length가 비슷함 → 더 단순하고 빠른 쪽 선택 가능.
- 실행 재현 안 됨 → seed/config/result가 모두 저장되는지 먼저 고침.

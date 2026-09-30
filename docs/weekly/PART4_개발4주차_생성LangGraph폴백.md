# PART 4 — 개발 4주차: 생성·Grounding·LangGraph·폴백

## 주간 목표
이번 주는 검색 결과를 실제 답변으로 연결하되, **생성 모델이 실패하거나 근거 없는 답변을 내더라도 Core RAG가 안전하게 폴백**하도록 만드는 주입니다.

흐름은 `retrieve → (근거 부족이면 fallback) → generate → grounding → (검증 실패면 fallback) → finalize` 로 고정합니다.

## 준비 라이브러리
- P0 Smoke: 별도 신규 의존성 없음. `ExtractiveGenerator` + 순수 Python `AgenticRAGWorkflow` 사용.
- 실제 생성: `ollama`, 로컬 Ollama 서버, 실제 설치 모델 1개.
- 그래프 승격: `langgraph`, `typing-extensions`.
- 기존 필수: `pytest`.

## Day 1 — 생성 프롬프트 계약
`src/intellectagent/generation.py`의 `build_grounded_prompt()`를 사용합니다.
- CONTEXT 밖 지식 금지
- 각 사실 주장 뒤 `[SOURCE_ID]`
- 근거 부족 시 정해진 폴백 문구
- 테스트에서는 Fake Ollama client를 주입해 서버 없이 계약 검증

실행:
```bash
PYTHONPATH=src python -m pytest tests/week4/test_generation_week4.py -q
```

## Day 2 — 순수 Python 노드 워크플로
`src/intellectagent/orchestration.py`
- `retrieve_node`
- `route_after_retrieve`
- `generate_node`
- `grounding_node`
- `route_after_grounding`
- `finalize_node`
- `fallback_node`

LangGraph가 없어도 이 워크플로 자체가 P0 기준선입니다.

## Day 3 — 양방향 폴백 테스트
자동 테스트는 정상 경로, 낮은 검색 점수 경로, 근거 없는 생성 경로를 모두 강제합니다.
```bash
PYTHONPATH=src python -m pytest tests/week4/test_orchestration.py -q
```

## Day 4 — LangGraph 연결
`src/intellectagent/optional_langgraph.py`는 위 노드 계약을 그대로 LangGraph에 연결합니다. LangGraph는 P0를 대체하지 않고 선택 실행기입니다.

실환경:
```bash
python scripts/week4/validate_langgraph.py
```

## Day 5 — Ollama Smoke
실제 로컬 모델을 설치한 뒤:
```bash
ollama list
python scripts/week4/validate_ollama.py --model qwen2.5:14b
```
통과 기준은 모델 호출 성공뿐 아니라 `validate_grounding()` 통과입니다.

## Day 6 — E2E와 Week 4 Gate
Smoke:
```bash
bash scripts/week4/run_week4_all.sh
```
Strict(실제 Ollama + LangGraph까지 설치 후):
```bash
PYTHONPATH=src python scripts/week4/run_week4_gate.py --strict
```

## 완료 Gate
- 전체 pytest 실패 0
- 정상 검색→생성→grounding→finalize 경로 PASS
- 근거 부족 폴백 경로 PASS
- 근거 없는 생성 차단 PASS
- 출처 ID는 실제 검색 hit에 존재
- LangGraph는 선택 의존성으로 격리
- 실제 Ollama 승격 시 grounding smoke PASS
- 실제 LangGraph 승격 시 graph.invoke PASS

## 막힐 때 복구
- Ollama 문제: `use_ollama=false`, ExtractiveGenerator 기준선 유지
- LangGraph 문제: `use_langgraph=false`, 순수 Python workflow 유지
- 생성이 인용 형식을 안 지킴: 프롬프트를 복잡하게 늘리지 말고 3회 이내 수정 후 해당 모델 후보를 보류
- 검색 점수가 불안정: 이번 주에는 retrieval 튜닝으로 돌아가지 말고 Week 3 기준선을 보존

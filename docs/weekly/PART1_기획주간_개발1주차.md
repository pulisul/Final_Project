# IntellectAgent AI 주간별 구현 가이드 — PART 1
## 범위: 기획 주간 + 개발 1주차

기준선: `IntellectAgent_Core_RAG_Gated_v2`. 원칙은 P0 Core RAG를 먼저 잠그고, 선택 기능은 기본 OFF 상태로 둔다.

## 0. 먼저 해야 할 일: 기준선 보존
```bash
cp -a IntellectAgent_Core_RAG_Gated_v2 IntellectAgent_Weekly_Build_v3
cd IntellectAgent_Weekly_Build_v3
python -m pytest -q
python scripts/run_release_gate.py --mode smoke
```
Smoke Gate가 PASS가 아니면 새로운 기능을 추가하지 않는다.

---

# 기획 주간

## 목표
코드를 많이 쓰는 주가 아니라 이후 코드의 계약을 고정하는 주다.

### 산출물 7종
1. `docs/planning/SCOPE.md` — P0/P1/P2
2. `docs/planning/DATA_SPEC.md` — 문서 형식·크기·개인정보 여부
3. `docs/planning/UI_FLOW.md`
4. `docs/planning/ERD.md`
5. `docs/planning/EVAL_PLAN.md`
6. `docs/planning/SECURITY_POLICY.md`
7. `docs/planning/RL_V01.md`

### P0 고정
- 안전한 문서 적재
- 청킹
- 임베딩
- vector store + metadata DB
- 검색
- 근거 기반 답변/폴백
- deterministic evaluation
- smoke/strict release gate

### P1/P2는 기본 OFF
`config/core.json`의 feature flag를 그대로 OFF로 유지한다.

## Day 1~2: 범위와 사용자 시나리오
예시 시나리오:
- 사용자가 허용된 문서 폴더를 인덱싱한다.
- 질문을 입력한다.
- 검색 근거가 충분하면 답변+출처를 받는다.
- 근거가 부족하면 폴백한다.
- 관리자는 평가와 Gate 결과를 확인한다.

## Day 3: UI 계약
UI보다 먼저 API 결과 형태를 고정한다.
예:
```python
{
  "answer": "...",
  "citations": ["chunk-id"],
  "fallback": False,
  "latency_ms": 123.4,
  "policy_version": "rule-v1"
}
```

## Day 4: 데이터/ERD
최소 테이블:
- documents
- chunks
- queries
- responses
- feedback
- policy_decisions

P0에서는 우선 documents/chunks를 실제 구현하고 나머지는 P1/P2용 계약으로 둔다.

데이터 인벤토리 생성:
```bash
python scripts/week1/inventory_data.py data/real_docs
```
생성 결과: `evidence/week1/data_inventory.json`

## Day 5: 모델/평가 계획
기본 후보:
- Embedder: `BAAI/bge-m3`
- Reranker: `BAAI/bge-reranker-v2-m3`
- Local LLM: Ollama의 실제 설치 가능한 후보 2개만 비교

평가 5문항은 반드시 `expected_doc_ids` 또는 정답 근거 chunk를 연결한다.

## Day 6: 보안/RL v0.1
보안:
- allowed root 밖 접근 거부
- symlink 거부
- 허용 확장자만 처리
- 최대 파일 크기 제한
- 개인정보 포함 여부 인벤토리에 수동 확인 필드 유지

RL v0.1:
- A0: top-k 기본 검색
- A1: 넓게 검색 후 rerank
- A2: query rewrite 후 rerank
- 폴백은 RL 행동에서 제외

## 기획 Gate
문서 7종이 존재하고, P0/P1/P2가 서로 침범하지 않으면 통과.

---

# 개발 1주차 — 문서 파이프라인

## 준비 라이브러리
Python은 3.11.x로 고정한다.

### 직접 의존성
- pypdf
- langchain-text-splitters
- sentence-transformers
- chromadb
- langchain
- langgraph
- langchain-ollama
- ollama
- ragas
- datasets
- numpy
- scikit-learn
- pandas
- plotly
- streamlit
- python-dotenv

### 개발 의존성
- pytest
- pytest-cov
- ruff

PyTorch는 OS/CPU/CUDA에 따라 공식 설치 절차로 별도 설치한다.

## Day 1 — Python 3.11 + 환경 기준선
```bash
cd IntellectAgent_Weekly_Build_v3
PYTHON_BIN=python3.11 bash scripts/week1/bootstrap_py311.sh
```
환경 기록:
```bash
python scripts/week1/check_environment.py
```
완료 조건:
- Python 3.11
- `pip check` 성공
- pytest 성공
- smoke gate 성공

## Day 2 — 안전한 문서 수집
현재 실제 구현은 `src/intellectagent/loaders.py`의 `SafeDocumentLoader`를 사용한다.

추가 테스트:
```bash
python -m pytest tests/test_security.py tests/test_loader_week1.py -q
```
검증 항목:
- root 외부 거부
- symlink 거부
- 확장자 거부
- 대용량 파일 거부
- 대소문자 확장자 처리

실문서 인벤토리:
```bash
python scripts/week1/inventory_data.py data/real_docs
```

## Day 3 — 의미 단위 청킹
기존 `chunk_document()`를 먼저 기준선으로 사용한다.
```bash
python -m pytest tests/test_chunking.py -q
```
임의 예외 규칙을 계속 늘리지 말고, 헤더/문단 기반 + 길이 폴백의 2단계만 유지한다.

## Day 4 — 길이 폴백 검증
경계 테스트를 추가한다.
- overlap >= size면 예외
- 빈 문서는 빈 chunk
- 긴 단일 문단은 길이 기준으로 분해
- 같은 문서 재실행 시 chunk ID 동일

## Day 5 — BGE-M3 smoke + 체크포인트
실제 BGE-M3를 켜기 전에 5개 문장만 검증:
```bash
python scripts/week1/validate_embeddings.py --model BAAI/bge-m3
```
확인:
- 벡터 개수=입력 개수
- 모든 벡터 차원 동일
- NaN/Inf 없음
- 0 벡터 없음

이후 `config/core.json`에서 `use_sentence_transformer=true`로 바꾸고 smoke gate를 다시 실행한다.
실패하면 즉시 false로 되돌린다.

## Day 6 — Chroma + SQLite
기본 원칙:
1. HashEmbedder + InMemory 상태의 smoke PASS를 보존한다.
2. Chroma만 ON한다.
3. 재인덱싱한다.
4. SQLite chunk ID와 Chroma ID의 일치 여부를 검사한다.

```bash
python -m intellectagent.cli --config config/core.json index
python scripts/run_release_gate.py --mode smoke
```

### 주간 Gate
```bash
python scripts/week1/run_week1_gate.py
```
통과 조건:
- 전체 pytest PASS
- Core smoke gate PASS
- 데이터 인벤토리 존재
- 환경 기록 존재
- 임베딩 smoke 기록 존재

일정 원문의 더 엄격한 최종 조건은 실데이터 기준으로 추가 확인한다:
- 지원 문서 처리율 >= 95%
- 실패 이유 기록
- NaN/0 벡터 0
- Chroma/SQLite ID 불일치 0
- 재실행 시 완료 파일 skip

## 실패 시 원칙
- BGE-M3 실패 -> HashEmbedder 기준선 유지
- Chroma 실패 -> InMemory 기준선 유지
- PDF 실패 -> 해당 파일만 failed에 기록하고 다른 문서는 계속
- 새로운 패키지 충돌 -> lock 갱신 금지
- 주간 Gate 실패 -> 개발 2주차 기능을 추가하지 않음

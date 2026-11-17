# PART 5 — 개발 5주차: Walking Skeleton · 통합 검증 · P0 동결

## 이번 주 목표
이번 주에는 새 기능을 추가하지 않는다. 지난 1~4주에 만든 Loader → Chunking → Embedding → Vector Search → Reranking → Generation → Grounding → Fallback 흐름을 처음부터 끝까지 한 번에 재현하고, P0 핵심 파일을 해시로 동결한다.

## Day 1 — 시연 시나리오와 Walking Skeleton
- `scripts/week5/run_walking_skeleton.py`
- 실제 문서 적재부터 질문 응답까지 한 번에 실행한다.
- 정상 질문과 근거 부족 질문을 최소 1개씩 포함한다.
- 산출물: `evidence/week5/WALKING_SKELETON.json`

## Day 2 — 첫 E2E 실행과 문제 수집
문제는 즉시 다 고치지 말고 먼저 분류한다.
- BLOCKER: 보안, 데이터 손실, Core 사용 불가
- CRITICAL: 우회책 없이 주요 P0 기능 실패
- MAJOR: 기능 저하가 있으나 우회 가능
- MINOR: UI·문구·비핵심 오류
- INFO: 개선 제안

명령 예:
```bash
python scripts/week5/classify_issues.py --title "문서 루트 탈출 가능" --component loader --security
```

## Day 3 — 버그 수정 원칙
1. BLOCKER/CRITICAL 먼저 처리한다.
2. 하루에 큰 버그 1~2개만 해결한다.
3. 수정 직후 해당 단위 테스트를 실행한다.
4. 전체 `pytest`와 Core Smoke Gate를 다시 실행한다.
5. MAJOR 이하 문제는 P0 제출을 막지 않으면 이슈로 남길 수 있다.

## Day 4 — 시연 리허설
실제 녹화가 아니라 다음 흐름만 확인한다.
1. 새 폴더에서 시작
2. 문서 적재
3. 정상 질문
4. 출처 표시
5. 근거 부족 질문의 폴백
6. 증거 JSON 확인

## Day 5 — P0 기능 동결
`freeze_p0.py`가 핵심 파일 SHA-256을 `P0_FREEZE.json`에 저장한다.
```bash
python scripts/week5/freeze_p0.py
python scripts/week5/verify_p0_freeze.py
```
동결 이후 P0 신규 기능 추가는 금지한다. BLOCKER/CRITICAL 수정으로 동결 파일을 변경한 경우 전체 회귀 테스트 후 다시 동결한다.

## Day 6 — 계획 재조정
토요일은 1시간 제한을 권장한다.
- 열린 이슈 수 확인
- P1/P2 일정 재조정
- 다음 주 평가는 P0 동결본을 대상으로 실행

## Week 5 Regression Suite
```bash
python scripts/week5/run_regression_suite.py
```
다음이 모두 PASS해야 한다.
- 전체 pytest
- Core Smoke Gate
- Week 4 Smoke Gate
- Walking Skeleton

## Week 5 Gate
```bash
python scripts/week5/run_week5_gate.py
```
Gate 조건:
- Regression Suite PASS
- P0 Freeze 존재
- Freeze 검증 PASS
- 열린 BLOCKER/CRITICAL 0건

## 완료 정의
- Walking Skeleton PASS
- 전체 pytest 0 failure
- Core/Week4 Smoke Gate PASS
- P0 핵심 파일 해시 동결
- BLOCKER/CRITICAL 0건
- 다음 주부터 신규 P0 기능 추가 금지

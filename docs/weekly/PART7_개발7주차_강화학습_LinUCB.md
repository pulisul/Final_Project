# PART 7 — 개발 7주차: Contextual Bandit 강화학습

## 목표
- 오프라인 보상과 실제 사용자 온라인 보상을 분리한다.
- 행동은 A0 기본검색, A1 넓은 검색+리랭킹, A2 질문 재작성+리랭킹 세 개로 고정한다.
- 폴백은 안전 규칙이며 RL 행동 공간에 포함하지 않는다.
- Random / Rule / LinUCB를 동일 synthetic offline environment에서 비교한다.
- 100회 smoke 후 500회 × 5 seed를 실행한다.
- 정책 저장/복원과 체크포인트 재시작을 검증한다.
- LinUCB가 규칙 정책보다 낫지 않으면 Rule 정책으로 롤백한다.

## 날짜별 작업
### 09/28
`RLLogDB`, `offline_reward()`, `online_reward()`를 구현한다. 두 보상은 DB와 보고서에서 혼합하지 않는다.

### 09/29
`RLState`, `OfflineBanditEnv`와 행동 3개를 고정한다. 상태는 질문 길이, top1 score, score gap, score std 네 개의 해석 가능한 특징부터 시작한다.

### 09/30
`RandomPolicy`, `RulePolicy`를 만든다. 규칙 정책은 RL이 반드시 이겨야 하는 현실적인 기준선이다.

### 10/01
`LinUCBPolicyV2`를 구현한다. A/b 행렬, UCB score, update, atomic JSON save/load를 포함한다.

### 10/02
`run_smoke100.py`로 100-step smoke를 실행한다. `verify_resume.py`로 100에서 중단한 실행을 200까지 재시작한 결과가 uninterrupted 200-step과 동일한지 확인한다.

### 10/03
`run_multiseed.py`로 500회 × 5 seed × 3정책을 실행한다. 정책별 seed 평균과 표준편차를 `policy_comparison.json`에 저장한다. LinUCB 평균이 Rule보다 낮으면 `selected_policy=rule`, `rollback_required=true`를 기록한다.

## 주간 Gate
- 전체 pytest PASS
- 3정책 비교 존재
- 각 정책 5 seed
- 평균·seed간 표준편차 존재
- LinUCB 저장/복원 동일성 PASS
- 체크포인트 재시작 동일성 PASS
- 자동 롤백 결정 생성

## 주의
현재 `OfflineBanditEnv`는 학습 알고리즘 구현을 검증하는 합성 환경이다. 이것을 실제 사용자 선호 학습 성능의 증거로 표현하면 안 된다. 8주차 사용자 피드백은 `online_reward()`로 별도로 축적하며, 온라인 업데이트 승격은 충분한 표본과 별도 Gate 이후에만 허용한다.

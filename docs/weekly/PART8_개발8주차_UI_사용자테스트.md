# PART 8 — 개발 8주차: Streamlit UI·캐시·피드백·사용자 테스트

## 목표
P0/P1/RL 기준선을 UI에 안전하게 노출한다. UI 오류가 Core RAG를 변경하지 않게 하고, 정책/인덱스 변경 시 캐시를 무효화하며, 동일 답변의 피드백 중복 저장과 개인정보 로그 노출을 막는다.

## 월요일 — 기본 화면과 장애 화면
- `app_streamlit.py` 작성
- 질문 입력, 답변, fallback 표시
- Core 로딩 예외를 사용자에게 명시적으로 표시
- `inspect_runtime_health()`로 config/policy/index 상태 분리

## 화요일 — 출처 카드와 정책 버전
- 검색 hit를 출처 카드로 표시
- chunk_id/doc_id/score/text 확인 가능
- `active_policy.json`을 읽어 현재 Rule/LinUCB/rollback 상태 표시

## 수요일 — 캐시와 세션
- 캐시 fingerprint = config SHA + policy SHA + index marker SHA
- 위 세 요소 중 하나가 바뀌면 Streamlit resource cache가 새로운 key로 로딩
- 세션 UUID는 브라우저 세션 동안 유지

## 목요일 — 스트리밍
- 답변 문자열을 UI에서 작은 토큰 단위로 `st.write_stream` 출력
- Core Generator 자체의 계약은 변경하지 않는다. 즉 UI 스트리밍 때문에 P0 동결 코드를 수정하지 않는다.

## 금요일 — 피드백과 사용자 테스트
- SQLite `feedback_log`에 UNIQUE feedback_key 적용
- 같은 session/question/response 조합의 중복 클릭은 두 번째 INSERT 거부
- 이유 자유기재란은 이메일/한국 휴대전화 패턴을 마스킹
- 사용자 테스트 양식: 2~3명 × 5질문, 답변 유용성/출처 유용성/폴백 적절성/이유/동의 여부

## 토요일 — Privacy/Regression Gate
- 전체 pytest
- 중복 피드백 테스트
- 캐시 무효화 테스트
- 개인정보 마스킹 테스트
- 사용자 테스트 템플릿 존재
- Smoke Gate PASS
- 실제 Streamlit 설치 환경에서는 `--strict-streamlit` 추가

## Gate
Smoke PASS는 Streamlit 패키지 없이도 Core/UI 로직을 검증한다. Strict PASS는 실제 Python 3.11 환경에서 Streamlit 설치와 app compile까지 요구한다.

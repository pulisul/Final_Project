# Changelog

## GitHub-ready snapshot — 2026-09-30

- Based on `IntellectAgent_Weekly_Build_v3_MAC_THREE_TIER`.
- Restored the latest direct dependencies used by `app_streamlit.py`.
- Applied the Ollama internal-knowledge answer/source fix.
- Applied the latest ChatGPT-style question input flow (`chat_input_flow_v2`).
- Kept Enter and **질문** button submission.
- Removed Python caches, local indexes, generated strict evidence, and intermediate runtime artifacts.
- Kept `artifacts/week7/active_policy.json` because runtime/release code references it.
- Added GitHub-safe `.gitignore`, submission documents, evidence examples, and upload guidance.

## Functional evolution

- Core RAG baseline
- Semantic chunk experiments
- BGE-M3 / reranker candidate path
- Ollama / LangGraph / grounding fallback
- Evaluation and RAGAS
- LinUCB policy experiments with rollback
- Streamlit UI
- Strict staged release pipeline
- macOS MPS profile
- Production config / E2E hardening
- Safe document upload
- RAG -> Web -> Ollama internal-knowledge fallback

## 2026-11-14 — 레드팀 수정과 설치형 배포
- 레드팀 하네스(scripts/redteam) 추가, 53개 시나리오. 보고서 docs/REDTEAM_REPORT.md
- 근거 검증: 내용 없는 인용·모델의 정직한 거절 구분, 서식(제목·굵은 머리말·출처 꼬리줄·전각 괄호)으로 인한 오거절 해소, 선택 숫자 검사
- textnorm: 질문·본문 NFC + 제로폭 문자 제거 / 이진 파일 위장 거부 / UTF-16 읽기 / 프롬프트 구획 표지 위조 방어
- 설치기(install.sh·install.bat·scripts/setup_env.py)와 실행기(run.sh·run.bat·manage.py) 추가, 예전 run_real_env.sh 는 호환 연결 스크립트로 변경

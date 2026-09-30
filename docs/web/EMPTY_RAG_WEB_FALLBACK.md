# Empty-RAG Web Fallback

## 목적

`data/real_docs`에 RAG 문서가 **0개**일 때만 인터넷 검색으로 질문에 답합니다.
문서가 1개 이상 있으면 기존 Production RAG(BGE-M3 → Chroma → BGE Reranker → Ollama)를 우선합니다.

## 동작

- 문서 0개: `DDGS 검색 → 검색결과 snippet → Ollama → Grounding → 웹 출처 URL`
- 문서 1개 이상: 기존 RAG 경로
- 웹 검색 실패: 임의 답변 금지, 오류/근거 없음 메시지
- 웹 결과 원문을 자동 다운로드하지 않음: 검색 결과의 제목/요약/URL만 근거로 사용

## 설정

`config/core.production.json`

- `use_empty_rag_web_fallback: true`
- `web_search_max_results: 5`
- `web_search_region: kr-kr`
- `web_search_safesearch: moderate`
- `web_search_timeout_seconds: 8`

기존 `use_web_search`는 별도의 allowlist 기반 Extension이므로 `false`를 유지합니다.

## 실행

```bash
bash scripts/mac/start_intellectagent.sh
```

이 스크립트는 `.venv-mac` 활성화, 환경변수 설정, Ollama 11434 점검/복구, `ddgs` 설치 확인 후 Streamlit을 실행합니다.

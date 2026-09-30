# Strict STEP 10~13 Runbook

## 전제
STEP 1~9가 실제 서버에서 PASS한 뒤 진행한다. STEP 10~13은 LLM·그래프·평가·UI 실행계층 검증이며 P0 코어를 수정하지 않는다.

## STEP 10 — Ollama grounded generation
1. `ollama --version`
2. `ollama list`
3. 필요 시 `ollama pull qwen2.5:14b`
4. `python scripts/strict/21_step10_ollama_grounded.py --model qwen2.5:14b`
PASS: 11434 접속, 모델 존재, 실제 응답, `[strict-c1]` citation 포함, fallback 아님.

## STEP 11 — LangGraph conditional graph
`python scripts/strict/22_step11_langgraph_conditional.py`
PASS: 정상 질문은 `route:finalize`, 빈 검색 결과는 `fallback_no_evidence`를 실제로 통과한다.

## STEP 12 — RAGAS
먼저 2문항 smoke:
`python scripts/strict/23_step12_ragas_strict.py --model qwen2.5:14b`
성공 후 20문항:
`python scripts/strict/23_step12_ragas_strict.py --model qwen2.5:14b --full`
RAGAS는 localhost Ollama만 사용하고 NetworkGuard 증거를 함께 남긴다.

## STEP 13 — Streamlit runtime
`python scripts/strict/24_step13_streamlit_strict.py`
임시 18501 포트에서 headless 앱을 띄워 실제 포트가 열리는지 확인한 뒤 프로세스를 종료한다.
브라우저 수동 확인은 이후 `streamlit run app_streamlit.py`로 수행한다.

## 일괄 Gate
`OLLAMA_MODEL=qwen2.5:14b bash scripts/strict/26_step10_13_all.sh`
20문항 RAGAS까지 포함하려면:
`OLLAMA_MODEL=qwen2.5:14b bash scripts/strict/26_step10_13_all.sh --full-ragas`

## 실패 순서
`STEP10_13_GATE.json`의 `blocked_by`만 먼저 고친다. Ollama가 실패한 상태에서 LangGraph/RAGAS/UI 오류를 동시에 추적하지 않는다.

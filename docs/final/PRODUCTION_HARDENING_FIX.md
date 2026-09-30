# Production Hardening Fix

이 수정본은 실제 ZIP 소스 리버싱에서 발견된 최종 Release 경로의 구조적 허점을 보완한다.

## 수정 핵심

1. `config/core.production.json`을 추가하여 smoke config와 production config를 분리했다.
2. production config는 `data/real_docs`, 별도 production Chroma/SQLite, BGE-M3, BGE reranker, Ollama를 사용한다.
3. `build_components()`가 `mac-native`, `linux-cuda`, `cpu` runtime profile의 실제 device를 BGE embedder와 reranker에 전달한다.
4. `CoreRAG`가 `candidate_k`를 실제 Retriever 호출에 전달한다.
5. `scripts/run_release_gate.py`가 `--config`, `--eval`, `--profile`을 받아 동일 production 기준선으로 Strict를 수행한다.
6. `scripts/strict/30_production_e2e.py`를 추가하여 실데이터가 실제로 Loader → BGE → Chroma → BGE reranker → Ollama → Grounding → LangGraph 경로를 통과하도록 했다.
7. STEP14는 `production_e2e` PASS를 필수로 하며 총 6단계를 동일한 production config에서 검증한다.
8. STEP15는 production E2E evidence, production config contract, 비어 있지 않은 real-data manifest가 없으면 Promotion을 거부한다.
9. Candidate benchmark는 더 이상 `Hit Rate >= 0`만으로 PASS하지 않으며 release policy의 Hit Rate/MRR threshold를 강제한다.
10. Chroma의 실제 chunk ID와 SQLite metadata chunk ID를 Strict에서 비교하도록 보강했다.
11. Streamlit은 `INTELLECTAGENT_CONFIG`와 `RUNTIME_PROFILE`을 사용하여 Mac production stack을 동일 config로 실행한다.
12. Release identity에 `core.production.json`과 `30_production_e2e.json`을 포함한다.

## 검증 결과

- 전체 테스트: 74/74 PASS
- Week 9 Smoke Gate: PASS
- Smoke config를 production STEP14에 주입: `production_config_mismatch`로 차단됨
- Production config contract: PASS
- 현재 검증 컨테이너의 production STEP14: `preflight`에서 정상 차단
- STEP14 실패 상태에서 STEP15 Promotion: 정상 거부
- Python compileall: PASS
- 모든 shell script `bash -n`: PASS

현재 컨테이너에는 실제 Mac MPS 환경, Python 3.11 production dependency lock, Ollama/BGE runtime, 실제 `data/real_docs`가 없으므로 최종 `PROMOTED`를 임의로 만들지 않았다. 실제 Mac에서 `scripts/mac/04_mac_full_pipeline.sh`를 실행해 production evidence를 채워야 한다.

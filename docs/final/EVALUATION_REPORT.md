# 평가 보고서

## 평가 층
1. Retrieval: Hit Rate@k, MRR, precision/recall 계열
2. Generation: citation success/validity, grounding
3. Fallback: fallback accuracy
4. Performance: mean/P95 latency
5. Human: 정확성, 근거성, 만족도, 출처 유용성, 가독성
6. Policy: Random vs Rule vs LinUCB, 5 seeds 평균/표준편차

## 현재 Smoke에서 확인된 핵심
- 평가 파이프라인과 원본 로그 저장이 동작한다.
- LinUCB는 Random보다 높았지만 Rule보다 낮아 자동 rollback이 수행되었다.
- Smoke 수치는 HashEmbedder/샘플 문서 기반이므로 실제 제출 성능으로 주장하지 않는다.

## Strict에서 남은 검증
실문서 + BGE-M3 + BGE reranker + Ollama + RAGAS local evaluator로 동일 평가 프로토콜을 재실행한다.

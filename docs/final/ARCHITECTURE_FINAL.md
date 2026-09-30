# 최종 시스템 아키텍처

```text
Documents -> Safe Loader -> Chunking -> Embedding -> Vector Store
                                      |                 |
                                      +-> Metadata DB <-+
User Question -> Retrieval -> Optional Reranker -> Generator -> Grounding Gate
                                                        |            |
                                                        | PASS       | FAIL
                                                        v            v
                                                     Answer       Fallback
                                                        |
                                                   Streamlit UI
                                                        |
                                              Feedback / Policy logs
                                                        |
                                           Random / Rule / LinUCB Eval
```

## 핵심 원칙
- P0 Core RAG는 LangGraph/RAGAS/RL/UI와 독립적으로 테스트 가능하다.
- 선택 기능 실패는 Core 실패로 전파하지 않는다.
- 모든 승격은 자동테스트와 Gate를 통과한 뒤에만 허용한다.
- 현재 smoke RL 결과에서는 Rule 정책이 LinUCB보다 우수하여 Rule을 활성화한다.

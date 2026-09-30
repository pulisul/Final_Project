# 최종 ERD

```text
documents 1 --- N chunks
queries   1 --- 1 responses
responses 1 --- N feedback_log
queries   1 --- N policy_decisions
```

핵심 메타데이터 DB와 RL/피드백 로그는 논리적으로 분리한다. 사용자 피드백 원문은 최소수집 원칙을 적용하고 질문/응답은 기본적으로 hash로 저장한다.

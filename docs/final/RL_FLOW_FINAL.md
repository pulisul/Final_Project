# 강화학습/정책 흐름

```text
State(question length, top1 score, score gap, score std)
 -> Policy(Random | Rule | LinUCB)
 -> A0 basic / A1 wide+rerank / A2 rewrite+rerank
 -> Offline reward
 -> LinUCB update
 -> Multi-seed comparison
 -> Promotion decision
 -> Rule if LinUCB <= Rule
```

Fallback은 RL 행동에서 제외하고 Grounding/Safety 계층에서 결정한다. 오프라인 보상과 실제 사용자 온라인 보상을 혼합해 같은 성능지표로 주장하지 않는다.

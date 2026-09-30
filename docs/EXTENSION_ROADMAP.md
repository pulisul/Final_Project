# Extension Roadmap

Core RAG is always releaseable. Extensions are never allowed to mutate the baseline until their candidate metrics pass `PromotionGate`.

## P1
1. grounding_verifier
2. incremental indexing (existing SHA checkpoint becomes the source of truth)
3. session_preferences
4. expanded human evaluation
5. regression dashboard

## P2
1. long_term_memory — explicit consent + delete-by-user required
2. finetune — adapter-only, isolated from base model
3. web_search — HTTPS + host allowlist + SSRF block
4. OCR — optional dependency, text-PDF path remains untouched
5. long-term online RL — starts in shadow/offline mode
6. multi_tool_agent — allowlisted tools + bounded steps

## Promotion rule
`candidate -> tests -> metrics -> PromotionGate -> feature flag ON`
If any gate fails, keep the last known-good baseline and leave the extension OFF.

# Delta merge record

GitHub-ready snapshot은 다음 순서로 구성했습니다.

1. Base: `IntellectAgent_Weekly_Build_v3_MAC_THREE_TIER.zip`
2. Dependency recovery: `IntellectAgent_DELTA_recover_app_dependencies.zip`
3. Ollama answer/source module fix: `IntellectAgent_DELTA_fix_ollama_answer_source.zip`
   - `ollama_knowledge.py` 반영
   - 해당 delta의 옛 `app_streamlit.py`는 최신 UI가 아니므로 최종 UI에 사용하지 않음
4. Final UI: `IntellectAgent_DELTA_chat_input_flow_v2.zip`

이전 `fix_source_keyerror`와 `chat_input_flow`은 후속 UI 수정본에 의해 superseded된 이력으로 취급합니다.

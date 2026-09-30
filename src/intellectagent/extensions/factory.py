from __future__ import annotations
from pathlib import Path
from . import ExtensionRegistry, GroundingVerifierExtension, SessionPreferencesExtension, LongTermMemoryExtension, OCRExtension, WebSearchExtension, LinUCBPolicy, FineTuneExtension, MultiToolAgentExtension

def build_extension_registry(cfg):
    f=cfg.feature_flags
    r=ExtensionRegistry()
    r.register(GroundingVerifierExtension(f.get('use_grounding_verifier',False)))
    r.register(SessionPreferencesExtension(cfg.index_dir.parent/'session_preferences.sqlite3',f.get('use_session_preferences',False)))
    r.register(LongTermMemoryExtension(cfg.index_dir.parent/'long_term_memory.sqlite3',f.get('use_long_term_memory',False)))
    r.register(OCRExtension(f.get('use_ocr',False)))
    r.register(WebSearchExtension(f.get('use_web_search',False),tuple(f.get('web_allowed_hosts',[])) if isinstance(f.get('web_allowed_hosts',[]),list) else ()))
    r.register(LinUCBPolicy(enabled=f.get('use_rl',False),state_path=cfg.project_root/'artifacts/linucb.json'))
    r.register(FineTuneExtension(f.get('use_finetune',False),None))
    r.register(MultiToolAgentExtension({},f.get('use_multi_tool_agent',False),max_steps=4))
    return r

from pathlib import Path
from intellectagent.ui_support import FeedbackStore, PolicyInfo, build_cache_fingerprint, load_policy_info, scrub_pii


def test_feedback_deduplicates(tmp_path):
    s=FeedbackStore(tmp_path/"f.db")
    p=PolicyInfo("rule","v1",True,"test")
    assert s.submit(session_id="s",question="q",response="a",thumb=1,policy=p)
    assert not s.submit(session_id="s",question="q",response="a",thumb=-1,policy=p)
    assert s.count()==1


def test_feedback_reason_scrubs_pii(tmp_path):
    s=FeedbackStore(tmp_path/"f.db")
    p=PolicyInfo("rule","v1",False,"")
    s.submit(session_id="s",question="q",response="a",thumb=-1,policy=p,reason="abc@example.com 010-1234-5678")
    reason=s.rows()[0]["reason"]
    assert "abc@example.com" not in reason
    assert "010-1234-5678" not in reason


def test_cache_changes_when_policy_changes(tmp_path):
    cfg=tmp_path/"config.json"; pol=tmp_path/"policy.json"
    cfg.write_text('{"a":1}')
    pol.write_text('{"active_policy":"rule"}')
    a=build_cache_fingerprint(config_path=cfg,policy_path=pol)
    pol.write_text('{"active_policy":"linucb"}')
    b=build_cache_fingerprint(config_path=cfg,policy_path=pol)
    assert a != b


def test_load_policy_info(tmp_path):
    p=tmp_path/"p.json"
    p.write_text('{"schema_version":1,"active_policy":"rule","rollback_applied":true,"reason":"lower_reward"}')
    x=load_policy_info(p)
    assert x.name=="rule"
    assert x.rollback_applied is True
    assert x.version.startswith("schema1-")


def test_scrub_pii():
    text=scrub_pii("email abc@example.com phone 010-1234-5678")
    assert "abc@example.com" not in text and "010-1234-5678" not in text

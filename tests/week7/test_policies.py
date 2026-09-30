from pathlib import Path
from intellectagent.rl_week7 import RLState, RulePolicy, LinUCBPolicyV2

def test_rule_policy_actions():
    p=RulePolicy()
    assert p.choose(RLState(50,.8,.2,.05)) == 0
    assert p.choose(RLState(50,.5,.02,.25)) == 1
    assert p.choose(RLState(50,.2,.02,.25)) == 2

def test_linucb_learns_and_roundtrips(tmp_path: Path):
    p=LinUCBPolicyV2(alpha=.1)
    s=RLState(80,.5,.02,.25)
    for _ in range(80):
        # Explicit supervised bandit update to verify policy mechanics.
        for a in range(3): p.update(a,s,1.0 if a==2 else 0.0)
    assert p.choose(s)==2
    f=tmp_path/'p.json'; p.save(f); q=LinUCBPolicyV2.load(f)
    assert q.choose(s)==2

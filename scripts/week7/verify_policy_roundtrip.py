from pathlib import Path
import json
from intellectagent.rl_week7 import LinUCBPolicyV2, RLState
p=LinUCBPolicyV2(alpha=.5)
s=RLState(80,.5,.03,.2)
for _ in range(20):
    a=p.choose(s); p.update(a,s,1.0 if a==1 else .1)
path=Path('evidence/week7/roundtrip_policy.json'); p.save(path)
q=LinUCBPolicyV2.load(path)
result={'before':p.choose(s),'after':q.choose(s),'pass':p.choose(s)==q.choose(s)}
Path('evidence/week7/policy_roundtrip.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(result)
raise SystemExit(0 if result['pass'] else 1)

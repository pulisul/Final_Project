from pathlib import Path
import json
src=Path('evidence/week7/policy_comparison.json')
d=json.loads(src.read_text(encoding='utf-8'))
decision=d['deployment']
out={
  'schema_version':1,
  'active_policy':decision['selected_policy'],
  'rollback_applied':bool(decision['rollback_required']),
  'source':'evidence/week7/policy_comparison.json',
  'reason':decision['reason'],
  'linucb_reward_mean':decision['linucb_reward_mean'],
  'rule_reward_mean':decision['rule_reward_mean'],
}
Path('artifacts/week7').mkdir(parents=True,exist_ok=True)
Path('artifacts/week7/active_policy.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))

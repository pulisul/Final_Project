from pathlib import Path
import json
from intellectagent.rl_week7 import RandomPolicy, RulePolicy, LinUCBPolicyV2, RLLogDB, simulate_policy, aggregate_runs, choose_deploy_policy

seeds=[2026100301,2026100302,2026100303,2026100304,2026100305]
out=Path('artifacts/week7/multiseed')
log=RLLogDB(Path('evidence/week7/rl_multiseed.sqlite3'))
rows=[]
for seed in seeds:
    rows.append(simulate_policy(RandomPolicy(seed), seed=seed, steps=500, out_dir=out, checkpoint_every=100, logger=log))
    rows.append(simulate_policy(RulePolicy(), seed=seed, steps=500, out_dir=out, checkpoint_every=100, logger=log))
    rows.append(simulate_policy(LinUCBPolicyV2(alpha=.8), seed=seed, steps=500, out_dir=out, checkpoint_every=100, logger=log))
report=aggregate_runs(rows)
report['deployment']=choose_deploy_policy(report, margin=0.0)
Path('evidence/week7').mkdir(parents=True,exist_ok=True)
Path('evidence/week7/policy_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report['deployment'],ensure_ascii=False,indent=2))

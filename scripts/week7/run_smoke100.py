from pathlib import Path
from intellectagent.rl_week7 import RandomPolicy, RulePolicy, LinUCBPolicyV2, RLLogDB, simulate_policy

out=Path('artifacts/week7/smoke100')
log=RLLogDB(Path('evidence/week7/rl_smoke.sqlite3'))
for p in [RandomPolicy(20261002), RulePolicy(), LinUCBPolicyV2(alpha=.8)]:
    s=simulate_policy(p, seed=20261002, steps=100, out_dir=out, checkpoint_every=25, resume=False, logger=log)
    print(p.name, round(s.reward_mean,4), s.action_counts)

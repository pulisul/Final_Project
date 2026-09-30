from pathlib import Path
import json, shutil
from intellectagent.rl_week7 import LinUCBPolicyV2, simulate_policy
root=Path('artifacts/week7/resume_test')
if root.exists(): shutil.rmtree(root)
# First run intentionally stops at 100, then resumes to 200.
simulate_policy(LinUCBPolicyV2(),seed=77,steps=100,out_dir=root,checkpoint_every=50,resume=False)
resumed=simulate_policy(LinUCBPolicyV2(),seed=77,steps=200,out_dir=root,checkpoint_every=50,resume=True)
# Compare with uninterrupted 200-step reference.
refroot=Path('artifacts/week7/resume_reference')
if refroot.exists(): shutil.rmtree(refroot)
reference=simulate_policy(LinUCBPolicyV2(),seed=77,steps=200,out_dir=refroot,checkpoint_every=50,resume=False)
passed=abs(resumed.reward_mean-reference.reward_mean)<1e-12 and resumed.action_counts==reference.action_counts and resumed.resumed
result={'pass':passed,'resumed':resumed.__dict__,'reference':reference.__dict__}
Path('evidence/week7/resume_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS' if passed else 'FAIL')
raise SystemExit(0 if passed else 1)

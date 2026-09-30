from pathlib import Path
from intellectagent.rl_week7 import RandomPolicy, RulePolicy, LinUCBPolicyV2, simulate_policy, aggregate_runs, choose_deploy_policy

def test_three_policy_report(tmp_path: Path):
    rows=[]
    for seed in [1,2]:
        rows += [
            simulate_policy(RandomPolicy(seed),seed=seed,steps=30,out_dir=tmp_path,resume=False),
            simulate_policy(RulePolicy(),seed=seed,steps=30,out_dir=tmp_path,resume=False),
            simulate_policy(LinUCBPolicyV2(),seed=seed,steps=30,out_dir=tmp_path,resume=False),
        ]
    report=aggregate_runs(rows)
    assert set(report['policies'])=={'random','rule','linucb'}
    dep=choose_deploy_policy(report)
    assert dep['selected_policy'] in {'rule','linucb'}

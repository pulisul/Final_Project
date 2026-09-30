from pathlib import Path
import json, subprocess, sys

def run(cmd):
    p=subprocess.run(cmd,text=True,capture_output=True)
    return {'pass':p.returncode==0,'returncode':p.returncode,'stdout':p.stdout[-4000:],'stderr':p.stderr[-4000:]}

checks={}
checks['pytest']=run([sys.executable,'-m','pytest','-q'])
checks['smoke100']=run([sys.executable,'scripts/week7/run_smoke100.py'])
checks['roundtrip']=run([sys.executable,'scripts/week7/verify_policy_roundtrip.py'])
checks['resume']=run([sys.executable,'scripts/week7/verify_resume.py'])
checks['multiseed']=run([sys.executable,'scripts/week7/run_multiseed.py'])
comparison=Path('evidence/week7/policy_comparison.json')
if comparison.exists():
    d=json.loads(comparison.read_text(encoding='utf-8'))
    pol=d.get('policies',{})
    checks['three_policies']={'pass':all(x in pol for x in ['random','rule','linucb'])}
    checks['five_seeds']={'pass':all(int(pol[x].get('seed_count',0))==5 for x in ['random','rule','linucb'])}
    checks['mean_std']={'pass':all('reward_mean' in pol[x] and 'reward_std_across_seeds' in pol[x] for x in ['random','rule','linucb'])}
    dep=d.get('deployment',{})
    checks['rollback_decision']={'pass':dep.get('selected_policy') in {'rule','linucb'},'details':dep}
else:
    for k in ['three_policies','five_seeds','mean_std','rollback_decision']:
        checks[k]={'pass':False,'reason':'missing comparison'}

overall=all(x.get('pass',False) for x in checks.values())
out={'schema_version':1,'week':7,'mode':'smoke','checks':checks,'overall_pass':overall}
Path('evidence/week7').mkdir(parents=True,exist_ok=True)
Path('evidence/week7/WEEK7_GATE.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'overall_pass':overall,'checks':{k:v.get('pass') for k,v in checks.items()}},ensure_ascii=False,indent=2))
raise SystemExit(0 if overall else 1)

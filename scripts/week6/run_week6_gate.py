#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ap=argparse.ArgumentParser(); ap.add_argument('--strict-ragas',action='store_true'); a=ap.parse_args()

def run(cmd):
    p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
    return {'pass':p.returncode==0,'returncode':p.returncode,'stdout':p.stdout[-4000:],'stderr':p.stderr[-4000:]}
checks={}
checks['pytest']=run([sys.executable,'-m','pytest','-q'])
checks['eval20_exists']={'pass':(ROOT/'data/eval/week6/eval20.jsonl').exists()}
checks['baseline_summary']={'pass':(ROOT/'evidence/week6/baseline_summary.json').exists()}
checks['reranked_summary']={'pass':(ROOT/'evidence/week6/reranked_summary.json').exists()}
checks['baseline_raw']={'pass':(ROOT/'evidence/week6/baseline_raw.jsonl').exists()}
checks['reranked_raw']={'pass':(ROOT/'evidence/week6/reranked_raw.jsonl').exists()}
checks['comparison']={'pass':(ROOT/'evidence/week6/evaluation_comparison.json').exists()}
checks['human_eval_template']={'pass':(ROOT/'evidence/week6/human_eval.csv').exists()}
checks['network_guard']=run([sys.executable,'scripts/week6/network_guard_smoke.py'])
# Same dataset/seed requirement
try:
    bs=json.loads((ROOT/'evidence/week6/baseline_summary.json').read_text(encoding='utf-8'))
    rs=json.loads((ROOT/'evidence/week6/reranked_summary.json').read_text(encoding='utf-8'))
    checks['same_dataset_seed']={'pass': bs['run_config']['dataset_version']==rs['run_config']['dataset_version'] and bs['run_config']['seed']==rs['run_config']['seed'], 'baseline':bs['run_config'],'reranked':rs['run_config']}
    required_groups={'retrieval','generation','fallback','latency'}
    checks['metrics_separated']={'pass': required_groups.issubset(set(bs['metrics'])) and required_groups.issubset(set(rs['metrics']))}
except Exception as e:
    checks['same_dataset_seed']={'pass':False,'error':str(e)}; checks['metrics_separated']={'pass':False,'error':str(e)}
ragas=run([sys.executable,'scripts/week6/ragas_smoke.py'])
if a.strict_ragas:
    checks['ragas_import_smoke']=ragas
else:
    checks['ragas_import_smoke']={'pass':True,'status':'OPTIONAL_NOT_RELEASE_BLOCKING','observed':ragas}
overall=all(v.get('pass',False) for v in checks.values())
out={'mode':'strict-ragas' if a.strict_ragas else 'smoke','checks':checks,'overall_pass':overall}
path=ROOT/'evidence/week6'/('WEEK6_STRICT_GATE.json' if a.strict_ragas else 'WEEK6_GATE.json')
path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2)); raise SystemExit(0 if overall else 1)

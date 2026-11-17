#!/usr/bin/env python3
from __future__ import annotations
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'evidence/week5';E.mkdir(parents=True,exist_ok=True)

def run(cmd):
    p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
    return {'pass':p.returncode==0,'returncode':p.returncode,'stdout_tail':p.stdout[-2000:],'stderr_tail':p.stderr[-1000:]}

def count_blockers():
    p=E/'issues.jsonl'
    if not p.exists(): return 0
    n=0
    for line in p.read_text(encoding='utf-8').splitlines():
        try:
            d=json.loads(line); n += int(bool(d.get('blocks_release')) and d.get('status','open')!='closed')
        except Exception: pass
    return n

def main():
    checks={}
    checks['regression_suite']=run([sys.executable,'scripts/week5/run_regression_suite.py'])
    if not (E/'P0_FREEZE.json').exists():
        checks['freeze_create']=run([sys.executable,'scripts/week5/freeze_p0.py'])
    else:
        checks['freeze_create']={'pass':True,'skipped':True}
    checks['freeze_verify']=run([sys.executable,'scripts/week5/verify_p0_freeze.py'])
    blockers=count_blockers(); checks['release_blockers']={'pass':blockers==0,'count':blockers}
    checks['p0_feature_freeze']={'pass':(E/'P0_FREEZE.json').exists()}
    overall=all(v.get('pass',False) for v in checks.values())
    result={'week':5,'mode':'smoke','checks':checks,'overall_pass':overall,'policy':'No new P0 features after freeze; only blocker/critical fixes may modify frozen files, followed by re-freeze.'}
    (E/'WEEK5_GATE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2));raise SystemExit(0 if overall else 2)
if __name__=='__main__':main()

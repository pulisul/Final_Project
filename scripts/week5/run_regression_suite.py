#!/usr/bin/env python3
from __future__ import annotations
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'evidence/week5'; E.mkdir(parents=True,exist_ok=True)

def run(cmd):
    p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
    return {'pass':p.returncode==0,'returncode':p.returncode,'stdout_tail':p.stdout[-2000:],'stderr_tail':p.stderr[-1200:]}

def main():
    checks={
      'pytest': run([sys.executable,'-m','pytest','-q']),
      'core_smoke_gate': run([sys.executable,'scripts/run_release_gate.py','--mode','smoke']),
      'week4_smoke_gate': run([sys.executable,'scripts/week4/run_week4_gate.py']),
      'walking_skeleton': run([sys.executable,'scripts/week5/run_walking_skeleton.py']),
    }
    result={'checks':checks,'overall_pass':all(v['pass'] for v in checks.values())}
    (E/'REGRESSION_SUITE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2));raise SystemExit(0 if result['overall_pass'] else 2)
if __name__=='__main__':main()

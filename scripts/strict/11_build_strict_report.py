#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from intellectagent.strict_diagnostics import classify_stage_failure
from intellectagent.strict_validation import write_json

def load(p):
    try: return json.loads(p.read_text(encoding='utf-8'))
    except Exception: return None

def main():
    integrated=load(ROOT/'evidence/strict/STRICT_INTEGRATED.json') or {}
    diag=load(ROOT/'evidence/strict/10_runtime_diagnostics.json') or {}
    stage_failures=[]
    for name,res in (integrated.get('results') or {}).items():
        if res.get('skipped'): continue
        if not res.get('pass',False): stage_failures.append(classify_stage_failure(name,res))
    report={
        'overall_pass': bool(integrated.get('overall_pass',False)),
        'blocked_by': integrated.get('blocked_by'),
        'runtime_blockers': diag.get('blockers',[]),
        'runtime_warnings': diag.get('warnings',[]),
        'stage_failures': stage_failures,
        'next_action': None,
    }
    if report['overall_pass']:
        report['next_action']='run_release_promotion'
    elif report['blocked_by']=='preflight':
        report['next_action']='fix_python311_lock_and_pip_check_then_resume'
    elif report['blocked_by']:
        report['next_action']=f"fix_{report['blocked_by']}_then_resume"
    else:
        report['next_action']='run_integrated_strict'
    write_json(ROOT/'evidence/strict/STRICT_SUMMARY.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())

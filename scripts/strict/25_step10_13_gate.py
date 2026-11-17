#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def run(cmd,timeout=1800):
    t=time.perf_counter()
    try:
        p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=timeout)
        return {'pass':p.returncode==0,'returncode':p.returncode,'elapsed_s':round(time.perf_counter()-t,3),'stdout':p.stdout[-5000:],'stderr':p.stderr[-2500:]}
    except subprocess.TimeoutExpired as e:return {'pass':False,'timeout':True,'elapsed_s':round(time.perf_counter()-t,3)}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='qwen2.5:14b'); ap.add_argument('--full-ragas',action='store_true'); a=ap.parse_args()
    stages=[
      ('ollama_grounded',[sys.executable,'scripts/strict/21_step10_ollama_grounded.py','--model',a.model]),
      ('langgraph_conditional',[sys.executable,'scripts/strict/22_step11_langgraph_conditional.py']),
      ('ragas',[sys.executable,'scripts/strict/23_step12_ragas_strict.py','--model',a.model]+(['--full'] if a.full_ragas else [])),
      ('streamlit_runtime',[sys.executable,'scripts/strict/24_step13_streamlit_strict.py']),
    ]
    results={}; blocked=None
    for name,cmd in stages:
        if blocked: results[name]={'pass':False,'skipped':True,'blocked_by':blocked}; continue
        results[name]=run(cmd)
        if not results[name]['pass']: blocked=name
    ev={'overall_pass':all(x.get('pass') for x in results.values()),'blocked_by':blocked,'model':a.model,'results':results}
    out=ROOT/'evidence/strict/STEP10_13_GATE.json'; out.write_text(json.dumps(ev,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(ev,ensure_ascii=False,indent=2)); return 0 if ev['overall_pass'] else 2
if __name__=='__main__': raise SystemExit(main())

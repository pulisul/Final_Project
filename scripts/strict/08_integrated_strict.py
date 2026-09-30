#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from intellectagent.strict_validation import write_json

def run(cmd, timeout=900):
    t=time.perf_counter()
    try:
        p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=timeout)
        return {'pass':p.returncode==0,'returncode':p.returncode,'elapsed_s':round(time.perf_counter()-t,3),'stdout':p.stdout[-6000:],'stderr':p.stderr[-3000:]}
    except subprocess.TimeoutExpired as e:
        return {'pass':False,'timeout':True,'elapsed_s':round(time.perf_counter()-t,3),'stdout':(e.stdout or '')[-6000:] if isinstance(e.stdout,str) else '', 'stderr':(e.stderr or '')[-3000:] if isinstance(e.stderr,str) else ''}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='qwen2.5:14b'); ap.add_argument('--resume',action='store_true'); a=ap.parse_args()
    stages=[
        ('preflight',[sys.executable,'scripts/strict/00_preflight.py']),
        ('bge_embedding',[sys.executable,'scripts/strict/01_validate_bge_embedding.py']),
        ('chroma',[sys.executable,'scripts/strict/02_validate_chroma.py']),
        ('bge_reranker',[sys.executable,'scripts/strict/03_validate_bge_reranker.py']),
        ('ollama',[sys.executable,'scripts/strict/04_validate_ollama.py','--model',a.model]),
        ('langgraph',[sys.executable,'scripts/strict/05_validate_langgraph.py']),
        ('ragas',[sys.executable,'scripts/strict/06_validate_ragas.py','--model',a.model]),
        ('streamlit',[sys.executable,'scripts/strict/07_validate_streamlit.py']),
        ('core_strict',[sys.executable,'scripts/run_release_gate.py','--mode','strict']),
    ]
    previous={}
    prev_path=ROOT/'evidence/strict/STRICT_INTEGRATED.json'
    if a.resume and prev_path.exists():
        try: previous=json.loads(prev_path.read_text(encoding='utf-8')).get('results',{})
        except Exception: previous={}
    results={}; blocked_by=None
    for name,cmd in stages:
        if blocked_by is not None:
            results[name]={'pass':False,'skipped':True,'blocked_by':blocked_by}; continue
        prev=previous.get(name,{})
        if a.resume and prev.get('pass') is True:
            results[name]={**prev,'reused':True}
            continue
        res=run(cmd); results[name]=res
        if not res['pass']: blocked_by=name
    overall=all(v.get('pass',False) for v in results.values())
    report={'overall_pass':overall,'blocked_by':blocked_by,'results':results}
    write_json(ROOT/'evidence/strict/STRICT_INTEGRATED.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2)); return 0 if overall else 2
if __name__=='__main__': raise SystemExit(main())

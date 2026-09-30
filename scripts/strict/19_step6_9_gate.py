#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def run(name, cmd):
    p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
    return {'name':name,'pass':p.returncode==0,'returncode':p.returncode,'stdout':p.stdout[-5000:],'stderr':p.stderr[-3000:]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--profile',choices=['auto','mac-native','linux-cuda','cpu'],default='auto'); ap.add_argument('--allow-cpu',action='store_true'); ap.add_argument('--embedding-model',default='BAAI/bge-m3'); ap.add_argument('--reranker-model',default='BAAI/bge-reranker-v2-m3'); ap.add_argument('--root',default='data/sample_docs'); ap.add_argument('--eval',default='data/eval/week2/eval15.jsonl'); a=ap.parse_args()
    profile='cpu' if a.allow_cpu else a.profile; py=sys.executable; steps=[]
    seq=[
      ('platform_runtime',[py,'scripts/strict/14_step6_gpu_runtime.py','--profile',profile]),
      ('bge_m3',[py,'scripts/strict/15_step7_bge_m3_strict.py','--model',a.embedding_model,'--profile',profile]),
      ('chroma_persistence',[py,'scripts/strict/16_step8_chroma_persistence.py']),
      ('bge_reranker',[py,'scripts/strict/17_step9_bge_reranker_strict.py','--model',a.reranker_model,'--profile',profile]),
      ('candidate_benchmark',[py,'scripts/strict/18_step9_candidate_benchmark.py','--root',a.root,'--eval',a.eval,'--embedding-model',a.embedding_model,'--reranker-model',a.reranker_model,'--profile',profile]),
    ]
    for name,cmd in seq:
        r=run(name,cmd); steps.append(r)
        if not r['pass']: break
    out={'requested_profile':profile,'steps':steps,'blocked_by':next((x['name'] for x in steps if not x['pass']),None),'overall_pass':all(x['pass'] for x in steps) and len(steps)==5}
    p=ROOT/'evidence/strict/STEP6_9_GATE.json'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(out,ensure_ascii=False,indent=2)); return 0 if out['overall_pass'] else 2
if __name__=='__main__': raise SystemExit(main())

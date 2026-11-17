#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def run(limit,model):
    p=subprocess.run([sys.executable,'scripts/week6/run_ragas_local_ollama.py','--model',model,'--limit',str(limit)],cwd=ROOT,text=True,capture_output=True,timeout=1800)
    return {'pass':p.returncode==0,'returncode':p.returncode,'stdout':p.stdout[-6000:],'stderr':p.stderr[-3000:]}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='qwen2.5:14b'); ap.add_argument('--full',action='store_true'); a=ap.parse_args()
    smoke=run(2,a.model); ev={'stage':'ragas','smoke2':smoke,'full20':{'skipped':True},'pass':False}
    if smoke['pass'] and a.full:
        ev['full20']=run(20,a.model); ev['pass']=ev['full20']['pass']
    else: ev['pass']=smoke['pass'] and not a.full
    out=ROOT/'evidence/strict/23_ragas_strict.json'; out.write_text(json.dumps(ev,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(ev,ensure_ascii=False,indent=2)); return 0 if ev['pass'] else 2
if __name__=='__main__': raise SystemExit(main())

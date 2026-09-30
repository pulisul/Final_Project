#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from intellectagent.strict_validation import tcp_open, write_json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='qwen2.5:14b'); a=ap.parse_args()
    tcp,err=tcp_open('127.0.0.1',11434,1.0)
    r={'model':a.model,'port_11434':tcp,'port_error':err,'pass':False}
    if tcp:
        p=subprocess.run([sys.executable,'scripts/week4/validate_ollama.py','--model',a.model],cwd=ROOT,text=True,capture_output=True)
        r.update({'validator_returncode':p.returncode,'stdout':p.stdout[-5000:],'stderr':p.stderr[-2000:],'pass':p.returncode==0})
    write_json(ROOT/'evidence/strict/04_ollama.json',r); print(json.dumps(r,ensure_ascii=False,indent=2)); return 0 if r['pass'] else 2
if __name__=='__main__': raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from intellectagent.strict_validation import write_json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='qwen2.5:14b'); a=ap.parse_args()
    p=subprocess.run([sys.executable,'scripts/week6/run_ragas_local_ollama.py','--model',a.model,'--limit','2'],cwd=ROOT,text=True,capture_output=True)
    r={'returncode':p.returncode,'stdout':p.stdout[-6000:],'stderr':p.stderr[-2000:],'pass':p.returncode==0}
    write_json(ROOT/'evidence/strict/06_ragas.json',r); print(json.dumps(r,ensure_ascii=False,indent=2)); return 0 if r['pass'] else 2
if __name__=='__main__': raise SystemExit(main())

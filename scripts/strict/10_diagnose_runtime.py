#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from intellectagent.strict_diagnostics import diagnose
from intellectagent.strict_validation import write_json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='qwen2.5:14b'); ap.add_argument('--min-free-gb',type=float,default=20.0); a=ap.parse_args()
    r=diagnose(ROOT,model=a.model,min_free_gb=a.min_free_gb)
    write_json(ROOT/'evidence/strict/10_runtime_diagnostics.json',r)
    print(json.dumps(r,ensure_ascii=False,indent=2))
    return 0 if r['pass'] else 2
if __name__=='__main__': raise SystemExit(main())

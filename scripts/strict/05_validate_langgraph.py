#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
from importlib.util import find_spec
sys.path.insert(0,str(ROOT/'src'))
from intellectagent.strict_validation import write_json

def main():
    r={'installed':find_spec('langgraph') is not None,'pass':False}
    if r['installed']:
        p=subprocess.run([sys.executable,'scripts/week4/validate_langgraph.py'],cwd=ROOT,text=True,capture_output=True)
        r.update({'returncode':p.returncode,'stdout':p.stdout[-5000:],'stderr':p.stderr[-2000:],'pass':p.returncode==0})
    write_json(ROOT/'evidence/strict/05_langgraph.json',r); print(json.dumps(r,ensure_ascii=False,indent=2)); return 0 if r['pass'] else 2
if __name__=='__main__': raise SystemExit(main())

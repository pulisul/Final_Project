#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from intellectagent.strict_validation import write_json

def main():
    p=subprocess.run([sys.executable,'scripts/week8/validate_streamlit.py'],cwd=ROOT,text=True,capture_output=True)
    r={'returncode':p.returncode,'stdout':p.stdout[-5000:],'stderr':p.stderr[-2000:],'pass':p.returncode==0}
    write_json(ROOT/'evidence/strict/07_streamlit.json',r); print(json.dumps(r,ensure_ascii=False,indent=2)); return 0 if r['pass'] else 2
if __name__=='__main__': raise SystemExit(main())

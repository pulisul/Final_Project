#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from intellectagent.strict_validation import environment_snapshot, write_json

def main():
    snap=environment_snapshot(ROOT)
    p=subprocess.run([sys.executable,'-m','pip','check'],cwd=ROOT,text=True,capture_output=True)
    snap['pip_check']={'pass':p.returncode==0,'stdout':p.stdout[-4000:],'stderr':p.stderr[-2000:]}
    snap['python_311']=snap['python_major_minor']=='3.11'
    snap['pass']=snap['python_311'] and snap['requirements_lock_exists'] and snap['pip_check']['pass']
    out=ROOT/'evidence/strict/00_preflight.json'; write_json(out,snap)
    print(json.dumps(snap,ensure_ascii=False,indent=2)); return 0 if snap['pass'] else 2
if __name__=='__main__': raise SystemExit(main())

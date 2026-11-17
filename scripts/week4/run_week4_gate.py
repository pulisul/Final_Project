#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
EVID=ROOT/"evidence/week4"

def run(cmd):
    p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
    return {"pass":p.returncode==0,"returncode":p.returncode,"stdout_tail":p.stdout[-2500:],"stderr_tail":p.stderr[-1500:]}

def load(path):
    try:return json.loads(path.read_text(encoding="utf-8"))
    except Exception:return None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--strict",action="store_true"); args=ap.parse_args()
    EVID.mkdir(parents=True,exist_ok=True)
    checks={}
    checks["pytest"]=run([sys.executable,"-m","pytest","-q"])
    checks["e2e_smoke"]=run([sys.executable,"scripts/week4/run_e2e_smoke.py"])
    e2e=load(EVID/"e2e_smoke.json") or {}
    checks["normal_branch"]={"pass":bool(e2e.get("normal_path_pass"))}
    checks["fallback_branch"]={"pass":bool(e2e.get("fallback_path_pass"))}
    if args.strict:
        checks["ollama"]=run([sys.executable,"scripts/week4/validate_ollama.py"])
        checks["langgraph"]=run([sys.executable,"scripts/week4/validate_langgraph.py"])
    else:
        checks["ollama"]={"pass":True,"skipped":True,"reason":"strict only"}
        checks["langgraph"]={"pass":True,"skipped":True,"reason":"strict only"}
    overall=all(v.get("pass",False) for v in checks.values())
    result={"mode":"strict" if args.strict else "smoke","checks":checks,"overall_pass":overall}
    p=EVID/("WEEK4_STRICT_GATE.json" if args.strict else "WEEK4_GATE.json")
    p.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if overall else 1)
if __name__=="__main__": main()

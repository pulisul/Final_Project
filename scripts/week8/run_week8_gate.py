from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def run(cmd):
    p=subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return {"pass": p.returncode==0, "returncode": p.returncode, "stdout": p.stdout[-5000:], "stderr": p.stderr[-5000:]}

ap=argparse.ArgumentParser(); ap.add_argument("--strict-streamlit", action="store_true"); args=ap.parse_args()
checks={}
checks["pytest"] = run([sys.executable,"-m","pytest","-q"])
checks["ui_logic_smoke"] = run([sys.executable,"scripts/week8/run_ui_logic_smoke.py"])
checks["user_test_template"] = run([sys.executable,"scripts/week8/build_user_test_template.py"])
required = [ROOT/"app_streamlit.py", ROOT/"evidence/week8/ui_logic_smoke.json", ROOT/"evidence/week8/user_test.csv", ROOT/"artifacts/week7/active_policy.json"]
checks["artifacts"]={"pass":all(p.exists() for p in required),"files":{str(p.relative_to(ROOT)):p.exists() for p in required}}
if args.strict_streamlit:
    checks["streamlit"] = run([sys.executable,"scripts/week8/validate_streamlit.py"])
else:
    checks["streamlit"] = {"pass": True, "skipped": True, "reason": "strict_streamlit_not_requested"}
overall=all(v.get("pass",False) for v in checks.values())
result={"week":8,"mode":"strict" if args.strict_streamlit else "smoke","checks":checks,"overall_pass":overall}
out=ROOT/"evidence/week8"/("WEEK8_STRICT_GATE.json" if args.strict_streamlit else "WEEK8_GATE.json")
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(result,ensure_ascii=False,indent=2)); raise SystemExit(0 if overall else 1)

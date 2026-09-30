from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; out=ROOT/'evidence/week9/FINAL_REGRESSION.json'; out.parent.mkdir(parents=True,exist_ok=True)
p=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,text=True,capture_output=True)
r={'pass':p.returncode==0,'stdout':p.stdout[-6000:],'stderr':p.stderr[-2000:]}
out.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(r,ensure_ascii=False,indent=2)); raise SystemExit(0 if r['pass'] else 2)

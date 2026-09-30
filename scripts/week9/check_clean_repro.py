from __future__ import annotations
import json, platform, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'evidence/week9/CLEAN_REPRO_CHECK.json'; out.parent.mkdir(parents=True,exist_ok=True)
checks={}
checks['python_311']={'pass':sys.version_info[:2]==(3,11),'actual':platform.python_version(),'required':'3.11.x'}
for name,path in [('requirements','requirements.txt'),('readme','README.md'),('pyproject','pyproject.toml')]:
    p=ROOT/path; checks[name]={'pass':p.exists() and p.stat().st_size>0,'path':path}
lock=ROOT/'requirements-lock.txt'; checks['requirements_lock']={'pass':lock.exists() and lock.stat().st_size>0,'path':'requirements-lock.txt'}
p=subprocess.run([sys.executable,'-m','pip','check'],cwd=ROOT,text=True,capture_output=True)
checks['pip_check']={'pass':p.returncode==0,'stdout':p.stdout[-2000:],'stderr':p.stderr[-1000:]}
r={'overall_pass':all(x['pass'] for x in checks.values()),'checks':checks}
out.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(r,ensure_ascii=False,indent=2)); raise SystemExit(0 if r['overall_pass'] else 2)

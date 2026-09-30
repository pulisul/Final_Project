from __future__ import annotations
import importlib.util, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
checks = {
    "streamlit_installed": importlib.util.find_spec("streamlit") is not None,
    "app_exists": (ROOT / "app_streamlit.py").exists(),
}
if checks["streamlit_installed"]:
    p = subprocess.run([sys.executable, "-m", "py_compile", str(ROOT / "app_streamlit.py")], capture_output=True, text=True)
    checks["app_compiles"] = p.returncode == 0
    checks["compile_stderr"] = p.stderr[-2000:]
else:
    checks["app_compiles"] = False
    checks["compile_stderr"] = "streamlit_not_installed"
checks["pass"] = all(checks[k] for k in ("streamlit_installed", "app_exists", "app_compiles"))
out=ROOT/"evidence"/"week8"/"streamlit_smoke.json"; out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(checks, ensure_ascii=False, indent=2))
raise SystemExit(0 if checks["pass"] else 1)

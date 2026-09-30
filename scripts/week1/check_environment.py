from __future__ import annotations
import importlib.metadata as md
import json, platform, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evidence" / "week1" / "environment.json"
REQUIRED_PY = (3, 11)
DIRECT = [
    "pypdf", "langchain-text-splitters", "sentence-transformers", "chromadb",
    "langchain", "langgraph", "langchain-ollama", "ollama", "ragas", "datasets",
    "numpy", "scikit-learn", "pandas", "plotly", "streamlit", "python-dotenv",
]

def pkg_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None

def main() -> int:
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"], text=True, capture_output=True
    )
    payload = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "python_311": sys.version_info[:2] == REQUIRED_PY,
        "platform": platform.platform(),
        "packages": {name: pkg_version(name) for name in DIRECT},
        "pip_check_pass": pip_check.returncode == 0,
        "pip_check_stdout": pip_check.stdout.strip(),
        "pip_check_stderr": pip_check.stderr.strip(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    missing = [k for k, v in payload["packages"].items() if v is None]
    return 0 if payload["python_311"] and payload["pip_check_pass"] and not missing else 2

if __name__ == "__main__":
    raise SystemExit(main())

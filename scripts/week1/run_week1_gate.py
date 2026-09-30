from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / "evidence" / "week1"

def run(cmd: list[str]) -> dict:
    p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    return {"pass": p.returncode == 0, "returncode": p.returncode, "stdout": p.stdout[-5000:], "stderr": p.stderr[-3000:]}

def main() -> int:
    checks = {
        "pytest": run([sys.executable, "-m", "pytest", "-q"]),
        "smoke_gate": run([sys.executable, "scripts/run_release_gate.py", "--mode", "smoke"]),
    }
    inventory = EV / "data_inventory.json"
    env = EV / "environment.json"
    emb = EV / "embedding_smoke.json"
    checks["data_inventory"] = {"pass": inventory.exists(), "path": str(inventory)}
    checks["environment_record"] = {"pass": env.exists(), "path": str(env)}
    checks["embedding_smoke_record"] = {"pass": emb.exists(), "path": str(emb)}
    result = {"overall_pass": all(v["pass"] for v in checks.values()), "checks": checks}
    EV.mkdir(parents=True, exist_ok=True)
    (EV / "WEEK1_GATE.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["overall_pass"] else 2

if __name__ == "__main__":
    raise SystemExit(main())

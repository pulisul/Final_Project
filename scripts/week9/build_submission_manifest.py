from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from intellectagent.release_week9 import validate_submission_bundle
r=validate_submission_bundle(ROOT)
out=ROOT/'evidence/week9/SUBMISSION_MANIFEST.json'; out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(r,ensure_ascii=False,indent=2)); raise SystemExit(0 if r['pass'] else 2)

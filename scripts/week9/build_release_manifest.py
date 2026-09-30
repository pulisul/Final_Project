from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from intellectagent.release_week9 import build_release_manifest, verify_release_manifest
out=ROOT/'evidence/week9/RELEASE_MANIFEST.json'; out.parent.mkdir(parents=True,exist_ok=True)
m=build_release_manifest(ROOT,'IntellectAgent-Weekly-Build-v3-RC')
out.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
v=verify_release_manifest(ROOT,m)
(ROOT/'evidence/week9/RELEASE_MANIFEST_VERIFY.json').write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'manifest':str(out),'file_count':m['file_count'],'verify':v},ensure_ascii=False,indent=2))
raise SystemExit(0 if v['pass'] else 2)

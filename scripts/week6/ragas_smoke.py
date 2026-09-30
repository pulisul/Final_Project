#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from intellectagent.ragas_adapter import ragas_import_smoke, build_ragas_records
out=ragas_import_smoke()
raw=ROOT/'evidence/week6/reranked_raw.jsonl'
if raw.exists():
    recs=build_ragas_records(raw)[:2]
    out['record_count']=len(recs)
    out['record_keys']=sorted(recs[0]) if recs else []
    (ROOT/'evidence/week6/ragas_smoke_payload.json').write_text(json.dumps(recs,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'evidence/week6/ragas_import_smoke.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
raise SystemExit(0 if out.get('pass') else 2)

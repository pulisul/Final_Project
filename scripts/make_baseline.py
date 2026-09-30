#!/usr/bin/env python3
import json
from pathlib import Path
src=Path('evidence/core_metrics.json')
out=Path('evidence/baseline_metrics.json')
if not src.exists():
    raise SystemExit('run evaluation first: evidence/core_metrics.json missing')
out.write_text(json.dumps(json.loads(src.read_text()),ensure_ascii=False,indent=2),encoding='utf-8')
print(out)

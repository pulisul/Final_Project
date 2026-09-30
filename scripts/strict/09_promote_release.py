#!/usr/bin/env python3
from __future__ import annotations
import json, shutil, sys
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from intellectagent.strict_validation import sha256_file, write_json

def main():
    strict=ROOT/'evidence/strict/STRICT_INTEGRATED.json'
    if not strict.exists():
        print('missing STRICT_INTEGRATED.json'); return 2
    data=json.loads(strict.read_text(encoding='utf-8'))
    if not data.get('overall_pass'):
        print('strict gate not passed; release promotion refused'); return 2
    lock=ROOT/'requirements-lock.txt'
    release={
        'release_status':'PROMOTED',
        'promoted_at_utc':datetime.now(timezone.utc).isoformat(),
        'requirements_lock_sha256':sha256_file(lock) if lock.exists() else None,
        'strict_evidence_sha256':sha256_file(strict),
        'active_policy':json.loads((ROOT/'artifacts/week7/active_policy.json').read_text(encoding='utf-8')) if (ROOT/'artifacts/week7/active_policy.json').exists() else None,
    }
    out=ROOT/'artifacts/strict/RELEASE_PROMOTION.json'; write_json(out,release)
    print(json.dumps(release,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())

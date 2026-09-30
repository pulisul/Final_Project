#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from intellectagent.release_candidate import verify_freeze_manifest, write_json

def main():
    p=ROOT/'evidence/week5/P0_FREEZE.json'
    if not p.exists(): raise SystemExit('P0_FREEZE.json missing')
    r=verify_freeze_manifest(ROOT,json.loads(p.read_text(encoding='utf-8')))
    write_json(ROOT/'evidence/week5/P0_FREEZE_VERIFY.json',r)
    print(json.dumps(r,ensure_ascii=False,indent=2))
    raise SystemExit(0 if r['pass'] else 2)
if __name__=='__main__': main()

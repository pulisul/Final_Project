#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from intellectagent.production_validation import run_production_e2e
from intellectagent.final_release import atomic_write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='config/core.production.json')
    ap.add_argument('--eval',default='data/eval/week6/eval20.jsonl')
    ap.add_argument('--root',default='data/real_docs')
    ap.add_argument('--profile',choices=['auto','mac-native','linux-cuda','cpu'],default='auto')
    a=ap.parse_args()
    policy=json.loads((ROOT/'config/release_policy.json').read_text(encoding='utf-8'))
    result=run_production_e2e(config_path=ROOT/a.config,eval_path=ROOT/a.eval,runtime_profile=a.profile,policy=policy,expected_root=ROOT/a.root)
    out=ROOT/'evidence/strict/30_production_e2e.json'; atomic_write_json(out,result)
    print(json.dumps(result,ensure_ascii=False,indent=2)); return 0 if result.get('pass') else 2
if __name__=='__main__': raise SystemExit(main())

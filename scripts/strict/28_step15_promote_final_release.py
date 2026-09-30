#!/usr/bin/env python3
from __future__ import annotations
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from intellectagent.final_release import atomic_write_json, active_policy_summary, collect_metrics, load_json, release_identity, sha256_file


def main():
    step14=ROOT/'evidence/strict/STEP14_INTEGRATED_GATE.json'; prod_e2e=ROOT/'evidence/strict/30_production_e2e.json'; lock=ROOT/'requirements-lock.txt'
    if not step14.exists(): print('STEP14_INTEGRATED_GATE.json missing; promotion refused'); return 2
    gate=load_json(step14)
    if not gate.get('overall_pass'): print(f"STEP14 not passed (blocked_by={gate.get('blocked_by')}); promotion refused"); return 2
    if not prod_e2e.exists() or not load_json(prod_e2e).get('pass'): print('production E2E evidence missing/failed; promotion refused'); return 2
    if not lock.exists(): print('requirements-lock.txt missing; promotion refused'); return 2
    policy=active_policy_summary(ROOT)
    if not policy.get('pass'): print('active policy invalid; promotion refused'); return 2
    if gate.get('using_sample_data'): print('sample-data evidence cannot be promoted; promotion refused'); return 2
    if not gate.get('production_contract',{}).get('pass'): print('production config contract invalid; promotion refused'); return 2
    if not gate.get('data_manifest',{}).get('file_count'): print('real data manifest empty; promotion refused'); return 2

    ident=release_identity(ROOT)
    promotion={
      'schema_version':3,'runtime_profile':gate.get('runtime_profile','auto'),'release_status':'PROMOTED','release_id':ident['release_id'],
      'promoted_at_utc':datetime.now(timezone.utc).isoformat(),'requirements_lock_sha256':sha256_file(lock),'step14_sha256':sha256_file(step14),
      'production_config':gate.get('config'),'production_config_sha256':gate.get('config_sha256'),'eval_sha256':gate.get('eval_sha256'),
      'data_manifest_sha256':gate.get('data_manifest',{}).get('manifest_sha256'),'active_policy':policy,'evidence_hashes':ident['files'],
      'metrics':collect_metrics(ROOT),'production_e2e':load_json(prod_e2e).get('metrics',{}),
      'rollback_target':{'policy':'rule' if policy.get('active_policy')!='rule' else policy.get('active_policy'),'p0_freeze':'evidence/week5/P0_FREEZE.json'},
    }
    out=ROOT/'artifacts/strict/FINAL_RELEASE_PROMOTION.json'; atomic_write_json(out,promotion)
    reloaded=load_json(out); verified=reloaded.get('release_id')==promotion['release_id'] and reloaded.get('release_status')=='PROMOTED'
    verify={'pass':verified,'release_id':promotion['release_id'],'promotion_sha256':sha256_file(out)}
    atomic_write_json(ROOT/'evidence/strict/STEP15_PROMOTION_VERIFY.json',verify)
    import json; print(json.dumps({'promotion':promotion,'verification':verify},ensure_ascii=False,indent=2)); return 0 if verified else 2
if __name__=='__main__': raise SystemExit(main())

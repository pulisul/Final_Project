#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from intellectagent.config import CoreConfig
from intellectagent.final_release import atomic_write_json, compare_snapshots, collect_metrics, snapshot_paths, active_policy_summary, sha256_file
from intellectagent.production_validation import validate_production_contract, data_manifest


def run(name, cmd, timeout=7200):
    t=time.perf_counter()
    try:
        p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=timeout)
        return {'name':name,'pass':p.returncode==0,'returncode':p.returncode,'elapsed_s':round(time.perf_counter()-t,3),'stdout':p.stdout[-8000:],'stderr':p.stderr[-4000:]}
    except subprocess.TimeoutExpired as e:
        return {'name':name,'pass':False,'timeout':True,'elapsed_s':round(time.perf_counter()-t,3),'stdout':(e.stdout or '')[-8000:] if isinstance(e.stdout,str) else '', 'stderr':(e.stderr or '')[-4000:] if isinstance(e.stderr,str) else ''}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='config/core.production.json')
    ap.add_argument('--model',default=None)
    ap.add_argument('--root',default='data/real_docs')
    ap.add_argument('--eval',default='data/eval/week6/eval20.jsonl')
    ap.add_argument('--embedding-model',default=None)
    ap.add_argument('--reranker-model',default=None)
    ap.add_argument('--allow-sample',action='store_true',help='development only; never use for production promotion')
    ap.add_argument('--profile',choices=['auto','mac-native','linux-cuda','cpu'],default='auto')
    a=ap.parse_args()

    config_path=(ROOT/a.config).resolve(); cfg=CoreConfig.load(config_path)
    data_root=(ROOT/a.root).resolve(); eval_path=(ROOT/a.eval).resolve()
    model=a.model or cfg.ollama_model; embedding_model=a.embedding_model or cfg.embedding_model; reranker_model=a.reranker_model or cfg.reranker_model
    using_sample='sample_docs' in data_root.as_posix()
    before=snapshot_paths(ROOT); stages=[]; blocked=None

    contract=validate_production_contract(cfg,data_root)
    if not contract['pass']: blocked='production_config_mismatch'
    if using_sample and not a.allow_sample: blocked='real_data_required'
    if not data_root.exists(): blocked='data_root_missing'
    if not eval_path.exists(): blocked='eval_missing'

    commands=[]
    if blocked is None:
        commands=[
            ('preflight',[sys.executable,'scripts/strict/00_preflight.py']),
            ('step6_9',[sys.executable,'scripts/strict/19_step6_9_gate.py','--profile',a.profile,'--root',a.root,'--eval',a.eval,'--embedding-model',embedding_model,'--reranker-model',reranker_model]),
            ('step10_13',[sys.executable,'scripts/strict/25_step10_13_gate.py','--model',model,'--full-ragas']),
            ('production_e2e',[sys.executable,'scripts/strict/30_production_e2e.py','--config',a.config,'--profile',a.profile,'--root',a.root,'--eval',a.eval]),
            ('core_strict',[sys.executable,'scripts/run_release_gate.py','--mode','strict','--config',a.config,'--eval',a.eval,'--profile',a.profile]),
            ('final_regression',[sys.executable,'-m','pytest','-q']),
        ]

    for name,cmd in commands:
        if blocked is not None:
            stages.append({'name':name,'pass':False,'skipped':True,'blocked_by':blocked}); continue
        r=run(name,cmd); stages.append(r)
        if not r['pass']: blocked=name

    after=snapshot_paths(ROOT); baseline=compare_snapshots(before,after); policy=active_policy_summary(ROOT)
    if blocked is None and not baseline['pass']: blocked='baseline_changed_during_strict'
    if blocked is None and not policy['pass']: blocked='active_policy_invalid'
    dmanifest=data_manifest(data_root) if data_root.exists() else {'file_count':0,'manifest_sha256':None}
    ev={
      'schema_version':2,'stage':'step14_integrated_release_candidate','created_at_utc':datetime.now(timezone.utc).isoformat(),
      'config':a.config,'config_sha256':sha256_file(config_path),'model':model,'embedding_model':embedding_model,'reranker_model':reranker_model,'runtime_profile':a.profile,
      'data_root':a.root,'data_manifest':dmanifest,'eval':a.eval,'eval_sha256':sha256_file(eval_path) if eval_path.exists() else None,'using_sample_data':using_sample,
      'production_contract':contract,'baseline_integrity':baseline,'active_policy':policy,'stages':stages,
      'metrics':collect_metrics(ROOT),'blocked_by':blocked,
      'overall_pass':blocked is None and baseline['pass'] and policy['pass'] and len(stages)==6 and all(s.get('pass') for s in stages),
    }
    atomic_write_json(ROOT/'evidence/strict/STEP14_INTEGRATED_GATE.json',ev)
    print(json.dumps(ev,ensure_ascii=False,indent=2)); return 0 if ev['overall_pass'] else 2
if __name__=='__main__': raise SystemExit(main())

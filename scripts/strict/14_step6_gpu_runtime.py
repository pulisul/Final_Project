#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from intellectagent.platform_runtime import detect_runtime_profile, memory_snapshot, configure_mac_mps_fallback
from intellectagent.strict_gpu_validation import torch_cuda_snapshot, gpu_memory_snapshot, write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--profile',choices=['auto','mac-native','linux-cuda','cpu'],default='auto')
    ap.add_argument('--allow-cpu',action='store_true',help='legacy alias; forces cpu profile')
    a=ap.parse_args()
    profile=detect_runtime_profile('cpu' if a.allow_cpu else a.profile)
    configure_mac_mps_fallback(profile)
    r={
      'runtime_profile':profile.to_dict(),
      'memory':memory_snapshot(profile),
      'nvidia':gpu_memory_snapshot() if profile.accelerator=='cuda' else {'available':False,'not_required':True},
      'torch_cuda':torch_cuda_snapshot(require_cuda=profile.profile=='linux-cuda'),
      'pass':bool(profile.pass_runtime),
    }
    write_json(ROOT/'evidence/strict/14_gpu_runtime.json',r)
    print(json.dumps(r,ensure_ascii=False,indent=2)); return 0 if r['pass'] else 2
if __name__=='__main__': raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from intellectagent.platform_runtime import detect_runtime_profile, memory_snapshot, configure_mac_mps_fallback
from intellectagent.strict_gpu_validation import write_json


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='BAAI/bge-reranker-v2-m3'); ap.add_argument('--profile',choices=['auto','mac-native','linux-cuda','cpu'],default='auto'); a=ap.parse_args()
    profile=detect_runtime_profile(a.profile); configure_mac_mps_fallback(profile)
    r={'model':a.model,'runtime_profile':profile.to_dict(),'memory_before':memory_snapshot(profile),'pass':False}
    try:
        from sentence_transformers import CrossEncoder
        t0=time.perf_counter(); m=CrossEncoder(a.model,device=profile.torch_device); load_ms=(time.perf_counter()-t0)*1000
        pairs=[('RAG란 무엇인가요?','RAG는 검색과 생성 모델을 결합하는 방식입니다.'),('RAG란 무엇인가요?','사과는 붉은 과일입니다.')]
        t1=time.perf_counter(); vals=[float(x) for x in m.predict(pairs)]; infer_ms=(time.perf_counter()-t1)*1000
        finite=all(math.isfinite(x) for x in vals)
        r.update({'scores':vals,'finite':finite,'relevant_gt_irrelevant':vals[0]>vals[1],'model_load_ms':load_ms,'infer_2_ms':infer_ms,'memory_after':memory_snapshot(profile)})
        r['pass']=finite and vals[0]>vals[1]
    except Exception as e:r['error']=f'{type(e).__name__}: {e}'
    write_json(ROOT/'evidence/strict/17_bge_reranker_strict.json',r); print(json.dumps(r,ensure_ascii=False,indent=2)); return 0 if r['pass'] else 2
if __name__=='__main__': raise SystemExit(main())

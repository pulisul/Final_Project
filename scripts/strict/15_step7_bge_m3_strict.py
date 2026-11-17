#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from intellectagent.embeddings import SentenceTransformerEmbedder
from intellectagent.platform_runtime import detect_runtime_profile, memory_snapshot, configure_mac_mps_fallback
from intellectagent.strict_gpu_validation import finite_nonzero_vectors, write_json


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='BAAI/bge-m3'); ap.add_argument('--profile',choices=['auto','mac-native','linux-cuda','cpu'],default='auto'); a=ap.parse_args()
    profile=detect_runtime_profile(a.profile); configure_mac_mps_fallback(profile)
    rows=['RAG는 검색과 생성을 결합합니다.','사과는 과일입니다.','벡터 검색은 임베딩을 사용합니다.','문서 기반 질의응답','근거가 없으면 폴백합니다.']
    before=memory_snapshot(profile); r={'model':a.model,'runtime_profile':profile.to_dict(),'memory_before':before,'pass':False}
    try:
        t0=time.perf_counter(); emb=SentenceTransformerEmbedder(a.model,device=profile.torch_device); load_ms=(time.perf_counter()-t0)*1000
        t1=time.perf_counter(); vecs=emb.embed(rows); encode_ms=(time.perf_counter()-t1)*1000
        dims=sorted({len(v) for v in vecs}); after=memory_snapshot(profile)
        ok=len(vecs)==5 and len(dims)==1 and finite_nonzero_vectors(vecs)
        r.update({'count':len(vecs),'dims':dims,'finite_nonzero':finite_nonzero_vectors(vecs),'model_load_ms':load_ms,'encode_5_ms':encode_ms,'memory_after':after,'pass':ok})
    except Exception as e:r['error']=f'{type(e).__name__}: {e}'
    write_json(ROOT/'evidence/strict/15_bge_m3_strict.json',r); print(json.dumps(r,ensure_ascii=False,indent=2)); return 0 if r['pass'] else 2
if __name__=='__main__': raise SystemExit(main())

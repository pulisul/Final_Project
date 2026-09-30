#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from intellectagent.embeddings import SentenceTransformerEmbedder
from intellectagent.strict_validation import write_json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='BAAI/bge-m3'); a=ap.parse_args()
    rows=['RAG는 검색과 생성을 결합합니다.','사과는 과일입니다.','벡터 검색은 임베딩을 사용합니다.','문서 기반 질의응답','근거가 없으면 폴백합니다.']
    r={'model':a.model,'pass':False}
    try:
        vecs=SentenceTransformerEmbedder(a.model).embed(rows)
        dims={len(v) for v in vecs}; finite=all(math.isfinite(x) for v in vecs for x in v); nonzero=all(any(abs(x)>1e-12 for x in v) for v in vecs)
        r.update({'count':len(vecs),'dims':sorted(dims),'finite':finite,'nonzero':nonzero,'pass':len(vecs)==len(rows) and len(dims)==1 and finite and nonzero})
    except Exception as e: r['error']=f'{type(e).__name__}: {e}'
    write_json(ROOT/'evidence/strict/01_bge_embedding.json',r); print(json.dumps(r,ensure_ascii=False,indent=2)); return 0 if r['pass'] else 2
if __name__=='__main__': raise SystemExit(main())

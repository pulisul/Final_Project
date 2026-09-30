#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from intellectagent.strict_validation import write_json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='BAAI/bge-reranker-v2-m3'); a=ap.parse_args()
    r={'model':a.model,'pass':False}
    try:
        from sentence_transformers import CrossEncoder
        m=CrossEncoder(a.model)
        scores=m.predict([('RAG란 무엇인가요?','RAG는 검색과 생성을 결합하는 방식입니다.'),('RAG란 무엇인가요?','사과는 붉은 과일입니다.')])
        vals=[float(x) for x in scores]
        r.update({'scores':vals,'finite':all(math.isfinite(x) for x in vals),'relevant_gt_irrelevant':vals[0]>vals[1],'pass':all(math.isfinite(x) for x in vals) and vals[0]>vals[1]})
    except Exception as e: r['error']=f'{type(e).__name__}: {e}'
    write_json(ROOT/'evidence/strict/03_bge_reranker.json',r); print(json.dumps(r,ensure_ascii=False,indent=2)); return 0 if r['pass'] else 2
if __name__=='__main__': raise SystemExit(main())

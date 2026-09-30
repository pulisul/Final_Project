#!/usr/bin/env python3
from __future__ import annotations
import json, shutil, sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from intellectagent.vectorstore import ChromaVectorStore
from intellectagent.embeddings import HashEmbedder
from intellectagent.models import Chunk
from intellectagent.strict_validation import write_json

def main():
    r={'pass':False}
    tmp=Path(tempfile.mkdtemp(prefix='ia_chroma_'))
    try:
        store=ChromaVectorStore(str(tmp),'strict_smoke')
        chunks=[Chunk('c1','d1','RAG는 검색과 생성을 결합합니다.',{},'strict',0),Chunk('c2','d2','사과는 과일입니다.',{},'strict',0)]
        emb=HashEmbedder(); store.upsert(chunks,emb.embed([c.text for c in chunks]))
        hits=store.query(emb.embed(['RAG 검색'])[0],2)
        store.delete_doc('d2')
        r.update({'hit_count':len(hits),'top_doc_id':hits[0].doc_id if hits else None,'pass':len(hits)>=1})
    except Exception as e: r['error']=f'{type(e).__name__}: {e}'
    finally: shutil.rmtree(tmp,ignore_errors=True)
    write_json(ROOT/'evidence/strict/02_chroma.json',r); print(json.dumps(r,ensure_ascii=False,indent=2)); return 0 if r['pass'] else 2
if __name__=='__main__': raise SystemExit(main())

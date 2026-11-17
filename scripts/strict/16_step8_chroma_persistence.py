#!/usr/bin/env python3
from __future__ import annotations
import json, shutil, sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from intellectagent.models import Chunk
from intellectagent.vectorstore import ChromaVectorStore
from intellectagent.embeddings import HashEmbedder
from intellectagent.strict_gpu_validation import write_json


def main():
    temp=Path(tempfile.mkdtemp(prefix='ia-chroma-strict-')); r={'path':str(temp),'pass':False}
    try:
        chunks=[Chunk('c1','d1','RAG는 검색과 생성을 결합합니다.','semantic',0,{}),Chunk('c2','d2','사과는 과일입니다.','semantic',0,{})]
        emb=HashEmbedder(); vecs=emb.embed([c.text for c in chunks])
        s1=ChromaVectorStore(str(temp),'strict_persist'); s1.upsert(chunks,vecs)
        q=emb.embed(['RAG란 무엇인가요?'])[0]; first=s1.query(q,2)
        del s1
        s2=ChromaVectorStore(str(temp),'strict_persist'); second=s2.query(q,2)
        s2.delete_doc('d1'); after_delete=s2.query(q,10)
        r.update({
          'first_ids':[h.chunk_id for h in first],
          'reopened_ids':[h.chunk_id for h in second],
          'after_delete_doc_ids':[h.doc_id for h in after_delete],
          'reopen_preserved': [h.chunk_id for h in first]==[h.chunk_id for h in second],
          'delete_worked': all(h.doc_id!='d1' for h in after_delete),
        })
        r['pass']=bool(r['first_ids']) and r['reopen_preserved'] and r['delete_worked']
    except Exception as e: r['error']=f'{type(e).__name__}: {e}'
    finally:
        shutil.rmtree(temp,ignore_errors=True)
    write_json(ROOT/'evidence/strict/16_chroma_persistence.json',r); print(json.dumps(r,ensure_ascii=False,indent=2)); return 0 if r['pass'] else 2
if __name__=='__main__': raise SystemExit(main())

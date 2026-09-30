#!/usr/bin/env python3
from __future__ import annotations
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from intellectagent.config import CoreConfig
from intellectagent.factory import build_components
from intellectagent.loaders import SafeDocumentLoader
from intellectagent.metadata import MetadataDB
from intellectagent.indexing import Indexer
from intellectagent.orchestration import AgenticRAGWorkflow


def main():
    cfg=CoreConfig.load(ROOT/'config/core.json')
    emb,store,retr,gen,pipe=build_components(cfg)
    loader=SafeDocumentLoader(cfg.allowed_root,cfg.allowed_extensions,cfg.max_file_bytes)
    db=MetadataDB(ROOT/'evidence/week5/walking_metadata.sqlite3')
    idx=Indexer(loader,emb,store,db,cfg).index_all(ROOT/'evidence/week5/walking_checkpoint.json')
    flow=AgenticRAGWorkflow(retr,gen,top_k=cfg.top_k,fallback_min_score=cfg.fallback_min_score)
    cases=['RAG는 무엇인가요?','사과는 무엇인가요?']
    rows=[]
    for q in cases:
        t=time.perf_counter(); ans,state=flow.run(q); elapsed=(time.perf_counter()-t)*1000
        rows.append({'question':q,'fallback':ans.fallback,'citations':list(ans.citations),'trace':state.trace,'latency_ms':elapsed})
    result={'index':idx,'cases':rows,'pass':idx.get('success_ratio',0)>=0.95 and all('retrieve' in r['trace'] for r in rows)}
    p=ROOT/'evidence/week5/WALKING_SKELETON.json';p.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2));raise SystemExit(0 if result['pass'] else 2)
if __name__=='__main__':main()

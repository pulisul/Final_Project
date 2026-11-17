#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; SRC=ROOT/'src'
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from intellectagent.config import CoreConfig
from intellectagent.loaders import SafeDocumentLoader
from intellectagent.embeddings import HashEmbedder
from intellectagent.vectorstore import InMemoryVectorStore
from intellectagent.metadata import MetadataDB
from intellectagent.indexing import Indexer
from intellectagent.retrieval import Retriever, NoopReranker
from intellectagent.reranking import TokenOverlapReranker
from intellectagent.generation import ExtractiveGenerator
from intellectagent.pipeline import CoreRAG
from intellectagent.evaluation_week6 import Week6RunConfig, load_week6_cases, evaluate_pipeline_week6, save_run

ap=argparse.ArgumentParser(); ap.add_argument('--system',choices=['baseline','reranked'],required=True); ap.add_argument('--seed',type=int,default=20260921)
a=ap.parse_args(); random.seed(a.seed)
cfg=CoreConfig.load(ROOT/'config/core.json')
loader=SafeDocumentLoader(cfg.allowed_root,cfg.allowed_extensions,cfg.max_file_bytes)
embedder=HashEmbedder(); store=InMemoryVectorStore(); meta=MetadataDB(ROOT/'evidence/week6/eval_metadata.sqlite3')
Indexer(loader,embedder,store,meta,cfg).index_all(ROOT/'evidence/week6/eval_checkpoint.json')
reranker=NoopReranker() if a.system=='baseline' else TokenOverlapReranker()
pipe=CoreRAG(Retriever(embedder,store,reranker),ExtractiveGenerator(),cfg.top_k,cfg.fallback_min_score)
cases=load_week6_cases(ROOT/'data/eval/week6/eval20.jsonl')
run_id=f"week6-{a.system}-{int(time.time())}"
res=evaluate_pipeline_week6(pipe,cases,Week6RunConfig(run_id,a.seed,a.system,cfg.top_k,'week6-smoke-v1','sample-doc smoke only'))
save_run(res,summary_path=ROOT/f'evidence/week6/{a.system}_summary.json',raw_log_path=ROOT/f'evidence/week6/{a.system}_raw.jsonl')
print(json.dumps(res['metrics'],ensure_ascii=False,indent=2))

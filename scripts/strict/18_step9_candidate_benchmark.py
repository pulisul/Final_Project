#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, statistics, sys, tempfile, time, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from intellectagent.chunk_strategies import ChunkStrategyConfig, chunk_with_strategy
from intellectagent.embeddings import SentenceTransformerEmbedder
from intellectagent.evaluation import load_cases, hit_rate_at_k, mrr
from intellectagent.loaders import SafeDocumentLoader
from intellectagent.models import SearchHit
from intellectagent.retrieval import CrossEncoderReranker
from intellectagent.strict_gpu_validation import CandidateResult, choose_candidate, candidate_quality_pass, write_json
from intellectagent.platform_runtime import detect_runtime_profile, memory_snapshot, configure_mac_mps_fallback
from intellectagent.vectorstore import ChromaVectorStore


def p95(xs):
    return sorted(xs)[max(0,int(len(xs)*0.95)-1)] if xs else 0.0


def run_one(docs,cases,embedder,reranker,candidate_k,top_k,db_root):
    chunks=[c for d in docs for c in chunk_with_strategy(d,ChunkStrategyConfig(name='semantic'))]
    store=ChromaVectorStore(str(db_root/f'c{candidate_k}'),'bench')
    store.upsert(chunks,embedder.embed([c.text for c in chunks]))
    pairs=[]; total=[]; search=[]; rr=[]; rows=[]
    for case in cases:
        t_all=time.perf_counter(); t0=time.perf_counter()
        qv=embedder.embed([case.question])[0]; cand=store.query(qv,candidate_k); search_ms=(time.perf_counter()-t0)*1000
        t1=time.perf_counter(); ranked=reranker.rerank(case.question,cand); rr_ms=(time.perf_counter()-t1)*1000
        hits=ranked[:top_k]; total_ms=(time.perf_counter()-t_all)*1000
        pairs.append((case,hits)); total.append(total_ms); search.append(search_ms); rr.append(rr_ms)
        rows.append({'case_id':case.case_id,'hit_doc_ids':[h.doc_id for h in hits],'search_ms':search_ms,'rerank_ms':rr_ms,'total_ms':total_ms})
    return {
      'candidate_k':candidate_k,'top_k':top_k,'chunk_count':len(chunks),
      'metrics':{
        'hit_rate_at_k':hit_rate_at_k(pairs),'mrr':mrr(pairs),
        'search_ms_mean':statistics.mean(search) if search else 0.0,
        'rerank_ms_mean':statistics.mean(rr) if rr else 0.0,
        'latency_ms_mean':statistics.mean(total) if total else 0.0,
        'latency_ms_p95':p95(total),'reranker_fallback_rate':0.0,
      },'cases':rows
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='data/sample_docs'); ap.add_argument('--eval',default='data/eval/week2/eval15.jsonl')
    ap.add_argument('--embedding-model',default='BAAI/bge-m3'); ap.add_argument('--reranker-model',default='BAAI/bge-reranker-v2-m3'); ap.add_argument('--top-k',type=int,default=5); ap.add_argument('--candidate-ks',default='5,10,20'); ap.add_argument('--profile',choices=['auto','mac-native','linux-cuda','cpu'],default='auto')
    a=ap.parse_args(); profile=detect_runtime_profile(a.profile); configure_mac_mps_fallback(profile); root=Path(a.root).resolve(); cases=load_cases(Path(a.eval)); loader=SafeDocumentLoader(root,('.md','.txt','.pdf'),20*1024*1024); docs=[loader.load(p) for p in loader.discover()]
    ks=[int(x) for x in a.candidate_ks.split(',') if x.strip()]; temp=Path(tempfile.mkdtemp(prefix='ia-strict-bench-'))
    out={'root':str(root),'eval':a.eval,'sample_warning':str(root).endswith('data/sample_docs'),'embedding_model':a.embedding_model,'reranker_model':a.reranker_model,'top_k':a.top_k,'candidate_ks':ks,'runtime_profile':profile.to_dict(),'memory_before':memory_snapshot(profile),'pass':False}
    try:
        embedder=SentenceTransformerEmbedder(a.embedding_model,device=profile.torch_device); reranker=CrossEncoderReranker(a.reranker_model,device=profile.torch_device); results=[]
        for k in ks: results.append(run_one(docs,cases,embedder,reranker,k,a.top_k,temp))
        choices=[CandidateResult(candidate_k=r['candidate_k'],**r['metrics']) for r in results]
        winner=choose_candidate(choices)
        policy=json.loads((ROOT/'config/release_policy.json').read_text(encoding='utf-8'))
        prod=policy.get('production',{})
        selected=next(r for r in results if r['candidate_k']==winner.candidate_k)
        quality_pass=candidate_quality_pass(winner,min_hit_rate_at_k=float(prod.get('candidate_min_hit_rate_at_k',0.0)),min_mrr=float(prod.get('candidate_min_mrr',0.0)))
        out.update({'results':results,'selected_candidate_k':winner.candidate_k,'selected_metrics':selected['metrics'],'quality_thresholds':{'min_hit_rate_at_k':prod.get('candidate_min_hit_rate_at_k',0.0),'min_mrr':prod.get('candidate_min_mrr',0.0)},'memory_after':memory_snapshot(profile),'pass':len(results)==len(ks) and quality_pass})
    except Exception as e: out['error']=f'{type(e).__name__}: {e}'
    finally: shutil.rmtree(temp,ignore_errors=True)
    write_json(ROOT/'evidence/strict/18_candidate_benchmark.json',out); print(json.dumps(out,ensure_ascii=False,indent=2)); return 0 if out['pass'] else 2
if __name__=='__main__': raise SystemExit(main())

from __future__ import annotations
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from .config import CoreConfig
from .evaluation import hit_rate_at_k, load_cases, mrr
from .factory import build_components
from .indexing import Indexer
from .loaders import SafeDocumentLoader
from .metadata import MetadataDB
from .optional_langgraph import build_graph

REQUIRED_PRODUCTION_FLAGS = (
    "use_chroma",
    "use_sentence_transformer",
    "use_reranker",
    "use_ollama",
    "use_langgraph",
    "use_ragas",
)


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def data_manifest(root: Path) -> dict[str, Any]:
    rows=[]
    if root.exists():
        for p in sorted(x for x in root.rglob('*') if x.is_file()):
            rows.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha256_file(p)})
    canonical=json.dumps(rows,sort_keys=True,separators=(',',':')).encode()
    return {'root':str(root),'file_count':len(rows),'files':rows,'manifest_sha256':hashlib.sha256(canonical).hexdigest()}


def _p95(xs:list[float])->float:
    return sorted(xs)[max(0,int(len(xs)*0.95)-1)] if xs else 0.0


def validate_production_contract(cfg: CoreConfig, expected_root: Path|None=None) -> dict[str, Any]:
    flags=cfg.feature_flags
    missing=[k for k in REQUIRED_PRODUCTION_FLAGS if not flags.get(k)]
    root_ok=True if expected_root is None else cfg.allowed_root.resolve()==expected_root.resolve()
    separate_index='production' in cfg.index_dir.parts and 'production' in cfg.metadata_db.parts
    return {
        'pass':not missing and root_ok and separate_index,
        'missing_required_flags':missing,
        'root_matches':root_ok,
        'allowed_root':str(cfg.allowed_root),
        'separate_production_index':separate_index,
        'candidate_k':cfg.candidate_k,
        'top_k':cfg.top_k,
    }


def run_production_e2e(*, config_path:Path, eval_path:Path, runtime_profile:str|None, policy:dict[str,Any], expected_root:Path|None=None)->dict[str,Any]:
    cfg=CoreConfig.load(config_path)
    contract=validate_production_contract(cfg,expected_root)
    out:dict[str,Any]={
        'config':str(config_path),
        'eval':str(eval_path),
        'contract':contract,
        'pass':False,
    }
    if not contract['pass']:
        out['error']='production_contract_invalid'; return out
    cases=load_cases(eval_path)
    if not cases:
        out['error']='empty_eval_set'; return out
    loader=SafeDocumentLoader(cfg.allowed_root,cfg.allowed_extensions,cfg.max_file_bytes)
    discovered=loader.discover()
    if not discovered:
        out['error']='no_real_documents'; return out
    embedder,store,retriever,generator,pipe=build_components(cfg,runtime_profile=runtime_profile)
    db=MetadataDB(cfg.metadata_db)
    checkpoint=cfg.evidence_dir/'production_e2e_index_checkpoint.json'
    idx=Indexer(loader,embedder,store,db,cfg).index_all(checkpoint)
    if idx.get('success_ratio',0.0)<0.95:
        out['error']='index_success_ratio_below_threshold'; out['index']=idx; return out

    graph=build_graph(core_rag=pipe)
    pairs=[]; fallback=[]; citations=[]; lat=[]; raw=[]
    for case in cases:
        t=time.perf_counter()
        result=graph.invoke({'question':case.question})
        elapsed=(time.perf_counter()-t)*1000
        ans=result.get('answer')
        if ans is None:
            out['error']=f'graph_returned_no_answer:{case.case_id}'; return out
        hits=list(ans.hits)
        pairs.append((case,hits)); fallback.append(float(ans.fallback==case.should_fallback)); citations.append(float(ans.fallback or bool(ans.citations))); lat.append(elapsed)
        raw.append({
          'case_id':case.case_id,'question':case.question,'fallback':ans.fallback,
          'citations':list(ans.citations),'hit_doc_ids':[h.doc_id for h in hits],
          'latency_ms':elapsed,'answer':ans.text,
          'trace':list(getattr(result.get('workflow_state'),'trace',[]) or []),
        })
    metrics={
      'count':len(cases),'hit_rate_at_k':hit_rate_at_k(pairs),'mrr':mrr(pairs),
      'fallback_accuracy':sum(fallback)/len(fallback),'citation_success':sum(citations)/len(citations),
      'latency_ms_mean':statistics.mean(lat),'latency_ms_p95':_p95(lat),
    }
    profile_name=(getattr(pipe,'runtime_profile',None).profile if getattr(pipe,'runtime_profile',None) is not None else (runtime_profile or cfg.runtime_profile or 'auto'))
    max_p95=policy['production']['max_latency_ms_p95_by_profile'].get(profile_name,policy['production']['max_latency_ms_p95_by_profile']['auto'])
    thresholds={
      'min_hit_rate_at_k':policy['production']['min_hit_rate_at_k'],
      'min_mrr':policy['production']['min_mrr'],
      'min_fallback_accuracy':policy['production']['min_fallback_accuracy'],
      'min_citation_success':policy['production']['min_citation_success'],
      'max_latency_ms_p95':max_p95,
    }
    passed=(metrics['hit_rate_at_k']>=thresholds['min_hit_rate_at_k'] and metrics['mrr']>=thresholds['min_mrr'] and metrics['fallback_accuracy']>=thresholds['min_fallback_accuracy'] and metrics['citation_success']>=thresholds['min_citation_success'] and metrics['latency_ms_p95']<=thresholds['max_latency_ms_p95'])
    out.update({
      'index':idx,'runtime_profile':profile_name,'metrics':metrics,'thresholds':thresholds,
      'raw_results':raw,'data_manifest':data_manifest(cfg.allowed_root),
      'eval_sha256':sha256_file(eval_path),'config_sha256':sha256_file(config_path),'pass':passed,
    })
    return out

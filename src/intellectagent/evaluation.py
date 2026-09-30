from __future__ import annotations
import json, statistics, time
from pathlib import Path
from .models import EvalCase

def hit_rate_at_k(cases_and_hits):
    vals=[]
    for case,hits in cases_and_hits:
        got={h.doc_id for h in hits}; vals.append(1.0 if got.intersection(case.relevant_doc_ids) else 0.0)
    return sum(vals)/len(vals) if vals else 0.0

def mrr(cases_and_hits):
    vals=[]
    for case,hits in cases_and_hits:
        rr=0.0
        for i,h in enumerate(hits,1):
            if h.doc_id in case.relevant_doc_ids: rr=1.0/i; break
        vals.append(rr)
    return sum(vals)/len(vals) if vals else 0.0

def load_cases(path:Path)->list[EvalCase]:
    out=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        d=json.loads(line); out.append(EvalCase(d["case_id"],d["question"],tuple(d.get("relevant_doc_ids",[])),bool(d.get("should_fallback",False))))
    return out

def evaluate_core(pipeline,cases:list[EvalCase])->dict:
    pairs=[]; lat=[]; fallback_ok=[]; citation_ok=[]
    for case in cases:
        t=time.perf_counter(); ans=pipeline.ask(case.question); lat.append((time.perf_counter()-t)*1000)
        pairs.append((case,list(ans.hits)))
        fallback_ok.append(float(ans.fallback==case.should_fallback))
        citation_ok.append(float(ans.fallback or bool(ans.citations)))
    return {
        "count":len(cases),"hit_rate_at_k":hit_rate_at_k(pairs),"mrr":mrr(pairs),
        "fallback_accuracy":sum(fallback_ok)/len(fallback_ok) if fallback_ok else 0.0,
        "citation_success":sum(citation_ok)/len(citation_ok) if citation_ok else 0.0,
        "latency_ms_mean":statistics.mean(lat) if lat else 0.0,
        "latency_ms_p95":sorted(lat)[max(0,int(len(lat)*0.95)-1)] if lat else 0.0,
    }

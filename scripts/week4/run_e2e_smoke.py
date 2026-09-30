#!/usr/bin/env python3
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))

from intellectagent.chunking import chunk_document
from intellectagent.embeddings import HashEmbedder
from intellectagent.generation import ExtractiveGenerator
from intellectagent.loaders import SafeDocumentLoader
from intellectagent.models import SearchHit
from intellectagent.orchestration import AgenticRAGWorkflow
from intellectagent.retrieval import Retriever
from intellectagent.vectorstore import InMemoryVectorStore


def main():
    loader=SafeDocumentLoader(ROOT/"data/sample_docs",(".md",".txt",".pdf"),10*1024*1024)
    embedder=HashEmbedder(); store=InMemoryVectorStore()
    docs=[]; chunks=[]
    for p in loader.discover():
        d=loader.load(p); docs.append(d)
        chunks.extend(chunk_document(d,120,1200,700,80))
    store.upsert(chunks,embedder.embed([c.text for c in chunks]))
    wf=AgenticRAGWorkflow(Retriever(embedder,store),ExtractiveGenerator(),top_k=3,fallback_min_score=0.12)
    questions=["RAG는 무엇인가요?","사과는 무엇인가요?"]
    rows=[]
    for q in questions:
        t=time.perf_counter(); answer,state=wf.run(q); elapsed=(time.perf_counter()-t)*1000
        rows.append({"question":q,"fallback":answer.fallback,"citations":list(answer.citations),"trace":state.trace,"latency_ms":elapsed})
    # Explicitly force the no-evidence branch so both directions are proven independent of sample embedding quirks.
    empty_wf=AgenticRAGWorkflow(type("R",(),{"search":lambda self,q,top_k,candidate_k=None: []})(),ExtractiveGenerator(),top_k=3)
    a,s=empty_wf.run("문서에 없는 질문")
    rows.append({"question":"forced-no-evidence","fallback":a.fallback,"citations":[],"trace":s.trace,"latency_ms":0.0})
    out={
        "document_count":len(docs),"chunk_count":len(chunks),"cases":rows,
        "normal_path_pass":any((not r["fallback"] and "route:finalize" in r["trace"]) for r in rows),
        "fallback_path_pass":any((r["fallback"] and "fallback_no_evidence" in r["trace"]) for r in rows),
    }
    out["overall_pass"]=out["normal_path_pass"] and out["fallback_path_pass"]
    path=ROOT/"evidence/week4/e2e_smoke.json"; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))
    raise SystemExit(0 if out["overall_pass"] else 1)

if __name__=="__main__": main()

#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from intellectagent.models import SearchHit
from intellectagent.generation import ExtractiveGenerator
from intellectagent.orchestration import AgenticRAGWorkflow
from intellectagent.optional_langgraph import build_graph
from intellectagent.strict_validation import write_json

class Retriever:
    def __init__(self,hits): self.hits=hits
    def retrieve(self,q,top_k=5): return self.hits[:top_k]

def run_case(hits):
    wf=AgenticRAGWorkflow(Retriever(hits),ExtractiveGenerator(),top_k=5,fallback_min_score=0.12)
    graph=build_graph(workflow=wf); out=graph.invoke({'question':'RAG란 무엇인가요?'})
    ws=out['workflow_state']; ans=out.get('answer')
    return {'trace':ws.trace,'fallback':bool(getattr(ans,'fallback',False)),'answer':getattr(ans,'text','')}

def main():
    ev={'stage':'langgraph_conditional','pass':False}
    try:
        good=run_case([SearchHit(chunk_id='c1',doc_id='d1',text='RAG는 검색과 생성을 결합합니다.',score=0.9,metadata={})])
        bad=run_case([])
        ev.update({'normal':good,'fallback':bad,'pass':('route:finalize' in good['trace'] and not good['fallback'] and 'fallback_no_evidence' in bad['trace'] and bad['fallback'])})
    except Exception as e: ev['error']=f'{type(e).__name__}: {e}'
    write_json(ROOT/'evidence/strict/22_langgraph_conditional.json',ev); print(json.dumps(ev,ensure_ascii=False,indent=2)); return 0 if ev['pass'] else 2
if __name__=='__main__': raise SystemExit(main())

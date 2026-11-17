#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from intellectagent.generation import ExtractiveGenerator
from intellectagent.models import SearchHit
from intellectagent.orchestration import AgenticRAGWorkflow

class R:
    def search(self,q,top_k,candidate_k=None):
        return [SearchHit("e"*20,"doc","RAG는 검색과 생성을 결합합니다.",0.9,{})]


def main():
    result={"import_pass":False,"invoke_pass":False,"error":None}
    try:
        from intellectagent.optional_langgraph import build_graph
        graph=build_graph(workflow=AgenticRAGWorkflow(R(),ExtractiveGenerator(),top_k=1,fallback_min_score=0.1))
        out=graph.invoke({"question":"RAG란?"})
        answer=out.get("answer")
        result.update({"import_pass":True,"invoke_pass":bool(answer and not answer.fallback),"trace":getattr(out.get("workflow_state"),"trace",[])})
    except Exception as e:
        result["error"]=f"{type(e).__name__}: {e}"
    result["overall_pass"]=result["import_pass"] and result["invoke_pass"]
    p=ROOT/"evidence/week4/langgraph_smoke.json"; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result["overall_pass"] else 1)

if __name__=="__main__": main()

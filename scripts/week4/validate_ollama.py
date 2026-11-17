#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from intellectagent.generation import OllamaGenerator
from intellectagent.grounding import validate_grounding
from intellectagent.models import SearchHit


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",default="qwen2.5:14b"); args=ap.parse_args()
    h=SearchHit("d"*20,"doc","RAG는 외부 문서를 검색하고 그 근거를 사용해 답변을 생성하는 방식입니다.",0.95,{})
    result={"model":args.model,"loaded":False,"grounding_pass":False,"error":None}
    try:
        text=OllamaGenerator(args.model).generate("RAG는 무엇인가요?",[h])
        g=validate_grounding(text,[h])
        result.update({"loaded":True,"answer":text,"grounding_pass":g.ok,"citation_coverage":g.citation_coverage})
    except Exception as e:
        result["error"]=f"{type(e).__name__}: {e}"
    result["overall_pass"]=result["loaded"] and result["grounding_pass"]
    p=ROOT/"evidence/week4/ollama_smoke.json"; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result["overall_pass"] else 1)

if __name__=="__main__": main()

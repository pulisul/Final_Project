#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from intellectagent.generation import OllamaGenerator
from intellectagent.models import SearchHit
from intellectagent.strict_services import tcp_probe, validate_grounded_text
from intellectagent.strict_validation import write_json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='qwen2.5:14b'); ap.add_argument('--host',default='127.0.0.1'); ap.add_argument('--port',type=int,default=11434); a=ap.parse_args()
    port=tcp_probe(a.host,a.port,2.0)
    evidence={'stage':'ollama_grounded','model':a.model,'port':port,'pass':False}
    if not port['pass']:
        evidence['error']='ollama_service_unreachable'; write_json(ROOT/'evidence/strict/21_ollama_grounded.json',evidence); print(json.dumps(evidence,ensure_ascii=False,indent=2)); return 2
    try:
        p=subprocess.run(['ollama','list'],text=True,capture_output=True,timeout=15)
        evidence['ollama_list_rc']=p.returncode; evidence['ollama_list_tail']=p.stdout[-3000:]
        if p.returncode!=0 or a.model.split(':')[0] not in p.stdout:
            evidence['error']='ollama_model_missing'; write_json(ROOT/'evidence/strict/21_ollama_grounded.json',evidence); print(json.dumps(evidence,ensure_ascii=False,indent=2)); return 2
        hit=SearchHit(chunk_id='strict-c1',doc_id='strict-doc',text='RAG는 검색과 생성 모델을 결합해 문서 근거를 활용하는 방식입니다.',score=0.99,metadata={})
        gen=OllamaGenerator(a.model)
        t=time.perf_counter(); text=gen.generate('RAG란 무엇인가요?',[hit]); elapsed=round((time.perf_counter()-t)*1000,3)
        grounded=validate_grounded_text(text,['strict-c1'])
        evidence.update({'response':text,'latency_ms':elapsed,'grounded':grounded,'pass':grounded['pass'] and not grounded['fallback']})
    except Exception as e:
        evidence['error']=f'{type(e).__name__}: {e}'
    write_json(ROOT/'evidence/strict/21_ollama_grounded.json',evidence); print(json.dumps(evidence,ensure_ascii=False,indent=2)); return 0 if evidence['pass'] else 2
if __name__=='__main__': raise SystemExit(main())

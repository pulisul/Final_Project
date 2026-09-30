#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, statistics, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from intellectagent.network_guard import LocalOnlyNetworkGuard

ap=argparse.ArgumentParser()
ap.add_argument('--model',default='qwen2.5:14b')
ap.add_argument('--raw',default='evidence/week6/reranked_raw.jsonl')
ap.add_argument('--limit',type=int,default=2)
a=ap.parse_args()

try:
    from openai import OpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness, ContextRecall
except Exception as e:
    print(json.dumps({'pass':False,'stage':'import','error':f'{type(e).__name__}: {e}'},ensure_ascii=False,indent=2))
    raise SystemExit(2)

rows=[json.loads(x) for x in (ROOT/a.raw).read_text(encoding='utf-8').splitlines() if x.strip()][:a.limit]
if not rows:
    raise SystemExit('no rows')
client=OpenAI(api_key='ollama',base_url='http://localhost:11434/v1')
llm=llm_factory(a.model,provider='openai',client=client)
faith=Faithfulness(llm=llm); recall=ContextRecall(llm=llm)
results=[]
with LocalOnlyNetworkGuard() as guard:
    for row in rows:
        if row.get('fallback'):
            continue
        contexts=row.get('retrieved_contexts') or []
        f=faith.score(user_input=row['question'],response=row['answer'],retrieved_contexts=contexts)
        r=recall.score(user_input=row['question'],retrieved_contexts=contexts,reference=row.get('reference_answer',''))
        results.append({'case_id':row['case_id'],'faithfulness':float(f.value),'context_recall':float(r.value)})

evidence={
    'pass':bool(results),
    'model':a.model,
    'evaluated_count':len(results),
    'means':{
        'faithfulness':statistics.mean(x['faithfulness'] for x in results) if results else None,
        'context_recall':statistics.mean(x['context_recall'] for x in results) if results else None,
    },
    'rows':results,
    'network_guard':guard.evidence(),
    'note':'RAGAS collections API + local Ollama via OpenAI-compatible localhost endpoint.'
}
out=ROOT/'evidence/week6/ragas_local_smoke.json'; out.write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(evidence,ensure_ascii=False,indent=2)); raise SystemExit(0 if evidence['pass'] else 2)

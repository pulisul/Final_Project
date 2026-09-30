"""Populate stable doc_ids in sample eval after hashing sample docs."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from intellectagent.loaders import SafeDocumentLoader
loader=SafeDocumentLoader(ROOT/"data/sample_docs",(".md",),10_000_000)
ids={p.name:loader.load(p).doc_id for p in loader.discover()}
rows=[
 {"case_id":"q1","question":"사과는 무엇인가요?","relevant_doc_ids":[ids["apples.md"]],"should_fallback":False},
 {"case_id":"q2","question":"검색 증강 생성 RAG는 문서를 어떻게 검색하고 근거를 생성하나요?","relevant_doc_ids":[ids["rag.md"]],"should_fallback":False},
 {"case_id":"q3","question":"화성기지 건설예산 금액은?","relevant_doc_ids":[],"should_fallback":True},]
with (ROOT/"data/eval/eval.jsonl").open("w",encoding="utf-8") as f:
    for r in rows:f.write(json.dumps(r,ensure_ascii=False)+"\n")
print(ids)

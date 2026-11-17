"""문서 폴더에서 평가용 질문 초안을 만든다 (qa_gen.py).

사용 (로컬 Ollama):
  PYTHONPATH=src python scripts/autorag/gen_qa.py --root data/real_docs --out evidence/qa_draft --model qwen2.5:14b --n 50 --unanswerable 10
산출물: qa_draft.jsonl(초안), qa_rejected.jsonl(걸러진 것과 사유), report.json(통계·설정)
검수: qa_draft.jsonl 에서 맞는 줄의 "status" 를 "approved" 로 바꾸고, 평가에는 qa_gen.load_approved() 결과만 쓴다.
--stub 은 Ollama 없이 배관만 확인하는 가짜 LLM 이다. 그 출력은 질문으로서 가치가 없다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from intellectagent.chunk_strategies import ChunkStrategyConfig, chunk_with_strategy
from intellectagent.lexical import BM25Index, tok_char_bigram
from intellectagent.loaders import SafeDocumentLoader
from intellectagent.qa_gen import generate_qa, generate_unanswerable, ollama_llm, oracle_filter, write_jsonl

from intellectagent.doc_parsers import OFFICE_EXTENSIONS

EXTS = (".md", ".txt", ".pdf", *sorted(OFFICE_EXTENSIONS))


def stub_llm(prompt: str) -> str:
    if prompt.startswith("당신은 문서 검색"):
        text = prompt.split("TEXT:\n", 1)[1].rsplit("\n\n질문:", 1)[0].strip()
        return f"{text[:14].strip()} 관련 내용은 무엇인가요?"
    if prompt.startswith("아래 TEXT만 근거로"):
        return prompt.split("TEXT:\n", 1)[1].split("\n", 1)[0][:80]
    return "다루지 않는 주제의 질문은 무엇인가요?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen2.5:14b"); ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--unanswerable", type=int, default=0); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-file-mb", type=int, default=60); ap.add_argument("--chunk-size", type=int, default=700); ap.add_argument("--chunk-overlap", type=int, default=80)
    ap.add_argument("--parse-cache", default=None, help="파싱 캐시 폴더(허용 루트 밖)"); ap.add_argument("--stub", action="store_true")
    ap.add_argument("--oracle", action="store_true", help="근거 텍스트만으로 정답이 복원되는지 LLM 으로 재검사해 못 하는 질문을 거른다(질문당 LLM 1회 추가)")
    ap.add_argument("--oracle-threshold", type=float, default=0.5)
    a = ap.parse_args()
    root, out = Path(a.root).resolve(), Path(a.out)
    loader = SafeDocumentLoader(root, EXTS, a.max_file_mb * 1024 * 1024)
    if a.parse_cache:
        from intellectagent.parse_cache import CachingDocumentLoader, ParseCache
        loader = CachingDocumentLoader(loader, ParseCache(Path(a.parse_cache)))
    docs, failed = [], []
    for p in loader.discover():
        try:
            docs.append(loader.load(p))
        except Exception as exc:  # 파싱 실패 문서는 건너뛰되 보고서에 남긴다
            failed.append({"path": str(p), "error": f"{type(exc).__name__}: {exc}"})
    cfg = ChunkStrategyConfig("length", a.chunk_size, a.chunk_overlap)
    chunks = [c for d in docs for c in chunk_with_strategy(d, cfg)]
    llm = stub_llm if a.stub else ollama_llm(a.model, seed=a.seed)
    res = generate_qa(llm, chunks, a.n, seed=a.seed)
    drafts, rejected = list(res.drafts), list(res.rejected)
    oracle_stats = None
    if a.oracle:
        drafts, rej_o, oracle_stats = oracle_filter(llm, drafts, threshold=a.oracle_threshold)
        rejected += rej_o
    if a.unanswerable:
        un = generate_unanswerable(llm, [d.metadata.get("name", d.doc_id) for d in docs], a.unanswerable,
                                   BM25Index(chunks, tok_char_bigram, idf_variant="lucene"))
        drafts += un.drafts; rejected += un.rejected
    if a.stub:
        for r in drafts:
            r.flags.append("stub_llm")
    write_jsonl(drafts, out / "qa_draft.jsonl"); write_jsonl(rejected, out / "qa_rejected.jsonl")
    report = {"documents": len(docs), "failed_documents": failed, "chunks": len(chunks), "stats": res.stats, "oracle": oracle_stats, "by_qtype": {t: sum(1 for r in drafts if r.qtype == t) for t in sorted({r.qtype for r in drafts})},
              "unanswerable_drafts": sum(r.should_fallback for r in drafts), "model": "stub" if a.stub else a.model, "seed": a.seed,
              "chunking": f"length-{a.chunk_size}-{a.chunk_overlap}", "note": "모든 항목은 status=draft. 검수 전에는 점수 근거로 쓰지 말 것."}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

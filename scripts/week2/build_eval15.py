from __future__ import annotations
import argparse
import json
from pathlib import Path

from intellectagent.loaders import SafeDocumentLoader


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/sample_docs")
    ap.add_argument("--out", default="data/eval/week2/eval15.jsonl")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    loader = SafeDocumentLoader(root, (".md", ".txt", ".pdf"), 10 * 1024 * 1024)
    docs = {p.name: loader.load(p) for p in loader.discover()}
    rag = docs.get("rag.md")
    apples = docs.get("apples.md")
    if not rag or not apples:
        raise SystemExit("sample mode requires rag.md and apples.md; replace this script's cases with real project questions for production")
    cases = [
        ("q01", "RAG는 무엇인가요?", [rag.doc_id], False, "fact"),
        ("q02", "RAG는 질문을 받은 뒤 어떤 순서로 처리하나요?", [rag.doc_id], False, "fact"),
        ("q03", "RAG에서 검색된 근거는 어디에 사용되나요?", [rag.doc_id], False, "fact"),
        ("q04", "이 프로젝트의 핵심 게이트가 요구하는 것은 무엇인가요?", [rag.doc_id], False, "fact"),
        ("q05", "검색 결과와 출처는 어떤 관계여야 하나요?", [rag.doc_id], False, "fact"),
        ("q06", "사과는 무엇인가요?", [apples.doc_id], False, "fact"),
        ("q07", "사과 문서는 어떤 테스트 목적에 사용되나요?", [apples.doc_id], False, "fact"),
        ("q08", "사과에 여러 품종이 있나요?", [apples.doc_id], False, "fact"),
        ("q09", "샘플 문서에서 과일로 언급된 것은 무엇인가요?", [apples.doc_id], False, "fact"),
        ("q10", "RAG 문서와 사과 문서는 각각 어떤 용도의 샘플인가요?", [rag.doc_id, apples.doc_id], False, "multi_doc"),
        ("q11", "RAG와 사과 샘플 문서가 공통으로 검증하려는 것은 무엇인가요?", [rag.doc_id, apples.doc_id], False, "multi_doc"),
        ("q12", "화성 기지의 건설 예산은 얼마인가요?", [], True, "fallback"),
        ("q13", "2028년 서울의 정확한 인구는 얼마인가요?", [], True, "fallback"),
        ("q14", "양자컴퓨터의 현재 시장 점유율을 알려주세요.", [], True, "fallback"),
        ("q15", "이 문서에 없는 특정 회사의 최신 매출을 알려주세요.", [], True, "fallback"),
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for cid, q, ids, fb, qtype in cases:
            f.write(json.dumps({
                "case_id": cid,
                "question": q,
                "relevant_doc_ids": ids,
                "should_fallback": fb,
                "question_type": qtype,
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(cases)} cases -> {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

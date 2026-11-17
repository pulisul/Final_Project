"""청킹 설정 x 검색 방식 격자 시연 (합성 한국어 코퍼스).

⚠ 성능 주장이 아니다. 정책 문장 14개를 3개의 긴 문서로 이어 붙인 합성 코퍼스이고, HashEmbedder 는 의미 임베딩이 아니다.
  보여 주는 것: (1) 같은 질문셋으로 청킹 설정을 바꿔 비교하는 파이프라인이 끝까지 돈다
               (2) 큰 청크=recall 유리 / 작은 청크=precision 유리라는 지표의 성격
  실제 결론은 실문서 + BGE-M3 + 검수된 질문셋으로 같은 스크립트를 다시 돌려 얻는다.
사용: PYTHONPATH=src:scripts/autorag python scripts/autorag/demo_chunk_grid.py [출력폴더]
"""
from __future__ import annotations

import sys
from pathlib import Path

from demo_grid import DOCS, LEXICAL, SEMANTIC
from intellectagent.chunk_grid import RetrieverSpec, build_candidates
from intellectagent.chunk_strategies import ChunkStrategyConfig
from intellectagent.grid_search import run_grid
from intellectagent.hashing import sha256_text
from intellectagent.models import Document
from intellectagent.qa_gen import QARecord

GROUPS = {"policyA": ["d01", "d04", "d08", "d12", "d13"], "policyB": ["d02", "d05", "d09", "d10", "d06"], "policyC": ["d03", "d07", "d11", "d14"]}


def main(out: Path | None):
    docs, member = [], {}
    for gid, ids in GROUPS.items():
        text = "\n\n".join(DOCS[i] for i in ids)
        docs.append(Document(gid, Path(f"{gid}.md"), text, sha256_text(text), {"name": f"{gid}.md"}))
        member.update({i: gid for i in ids})
    cases = [QARecord(f"q{n:02d}", q, (member[d],), gt_contents=(DOCS[d],), kind="lexical" if (q, d) in LEXICAL else "semantic")
             for n, (q, d) in enumerate(LEXICAL + SEMANTIC)]
    cfgs = [ChunkStrategyConfig("length", 100, 0), ChunkStrategyConfig("length", 250, 30), ChunkStrategyConfig("length", 800, 80)]
    specs = [RetrieverSpec("bm25"), RetrieverSpec("vector"), RetrieverSpec("hybrid_rrf"), RetrieverSpec("hybrid_cc", cc_weight=0.3)]
    metrics = ("retrieval_recall", "retrieval_token_recall", "retrieval_token_precision", "retrieval_token_f1")
    res = run_grid(build_candidates(docs, cfgs, specs), cases, top_k=2, metrics=metrics, strategy="mean", out_dir=out)
    print(f"{'candidate':42s} chunks  doc_recall  tok_recall  tok_prec  tok_f1")
    for r in res["rows"]:
        print(f"{r['candidate']:42s} {r['param_chunk_count']:5d}   {r['retrieval_recall']:8.3f}  {r['retrieval_token_recall']:9.3f}  "
              f"{r['retrieval_token_precision']:8.3f}  {r['retrieval_token_f1']:6.3f}")
    print("best by mean of metrics:", res["best"], "(합성 데이터 — 결론 아님)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else None)

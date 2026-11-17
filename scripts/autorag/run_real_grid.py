"""실문서 + 검수된 질문셋 + (BGE-M3) 로 '청킹 설정 x 검색 방식' 을 한 번에 비교한다.

사용 (팀 컴퓨터):
  PYTHONPATH=src python scripts/autorag/run_real_grid.py --root <문서폴더> --qa evidence/qa/qa_draft.jsonl --out evidence/grid
  # 배관만 확인: --embedder hash   (HashEmbedder: 의미 임베딩이 아니므로 vector 결과는 무의미)

전제
- qa_draft.jsonl 에서 검수한 줄의 "status" 를 "approved" 로 바꿔 둔다. approved 만 사용한다.
- gen_qa.py 를 돌린 *같은 문서 폴더* 를 --root 로 준다 (질문의 정답 문서 ID 는 파일 내용 해시에서 나온다).
- 답 없는 질문(should_fallback)은 검색 지표가 정의되지 않아 제외하고 개수만 보고한다.

해석할 때
- 청킹 설정끼리는 retrieval_token_* (정답 근거 텍스트 기준) 으로 비교한다. retrieval_recall 등 문서 단위 지표는 청킹에 둔감하다.
- 질문이 본문 단어를 그대로 베낀 경우(lexical_overlap 높음) BM25 가 유리하게 나온다. 그래서 overlap 높음/낮음을 나눠서도 보여 준다.
- 질문 수가 적으면(<30) 순위 차이는 우연일 수 있다. 보고서에 경고를 남긴다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from intellectagent.chunk_grid import RetrieverSpec, build_candidates
from intellectagent.chunk_strategies import ChunkStrategyConfig
from intellectagent.doc_parsers import OFFICE_EXTENSIONS
from intellectagent.embeddings import HashEmbedder, SentenceTransformerEmbedder
from intellectagent.grid_search import run_grid
from intellectagent.loaders import SafeDocumentLoader
from intellectagent.qa_gen import load_approved

EXTS = (".md", ".txt", ".pdf", *sorted(OFFICE_EXTENSIONS))
METRICS = ("retrieval_recall", "retrieval_token_recall", "retrieval_token_precision", "retrieval_token_f1")
MIN_RELIABLE = 30


def parse_chunks(text: str) -> list[ChunkStrategyConfig]:
    """'500:50' 또는 '500:50:ctx'(문맥 머리말 포함). 같은 크기를 머리말 유무로 나란히 비교할 수 있다."""
    out = []
    for part in text.split(","):
        bits = part.strip().split(":")
        out.append(ChunkStrategyConfig("length", int(bits[0]), int(bits[1]) if len(bits) > 1 and bits[1] else 0, context_header=len(bits) > 2 and bits[2] == "ctx"))
    return out


M3_ABLATION = ((1, 0, 0), (0, 1, 0), (2, 1, 0), (0.4, 0.2, 0.4))   # dense / sparse / dense+sparse / 전부


def parse_specs(text: str, cc_weight: float, m3_weights=(0.4, 0.2, 0.4), m3_colbert=True, m3_ablation=False) -> list[RetrieverSpec]:
    out: list[RetrieverSpec] = []
    for k in (x.strip() for x in text.split(",") if x.strip()):
        if k == "hybrid_cc":
            out.append(RetrieverSpec(k, cc_weight=cc_weight))
        elif k == "bge_m3":
            if m3_ablation:
                out += [RetrieverSpec("bge_m3", m3_weights=tuple(float(x) for x in w), m3_colbert=w[2] > 0) for w in M3_ABLATION]
            else:
                out.append(RetrieverSpec("bge_m3", m3_weights=tuple(float(x) for x in m3_weights), m3_colbert=m3_colbert))
        else:
            out.append(RetrieverSpec(k))
    return out


def star(res: dict) -> str:
    """★ 는 tok_f1 최고 후보(동점이면 목록 앞쪽). 지표 평균(mean) 선택은 recall 이 큰 청크에 유리해 청킹 비교 기준으로 부적절하다."""
    return max(res["rows"], key=lambda r: r["retrieval_token_f1"])["candidate"]


def table(title: str, res: dict) -> str:
    lines = [f"### {title}", "", "| 후보 | 청크수 | doc_recall | tok_recall | tok_precision | tok_f1 | ms/질문 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(res["rows"], key=lambda r: -r["retrieval_token_f1"]):
        mark = " ★" if r["candidate"] == star(res) else ""
        lines.append(f"| {r['candidate']}{mark} | {r['param_chunk_count']} | {r['retrieval_recall']:.3f} | {r['retrieval_token_recall']:.3f} | "
                     f"{r['retrieval_token_precision']:.3f} | {r['retrieval_token_f1']:.3f} | {r['latency_ms_mean']:.1f} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--qa", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--embedder", default="BAAI/bge-m3", help="모델 이름 또는 'hash'(배관 확인용)")
    ap.add_argument("--chunks", default="300:30,500:50,700:80,1000:100", help="size:overlap 쉼표 목록")
    ap.add_argument("--retrievers", default="vector,bm25,hybrid_rrf,hybrid_cc", help="vector,bm25,hybrid_rrf,hybrid_cc,bge_m3")
    ap.add_argument("--cc-weight", type=float, default=0.5)
    ap.add_argument("--m3-weights", type=float, nargs=3, default=[0.4, 0.2, 0.4], metavar=("DENSE", "SPARSE", "COLBERT"))
    ap.add_argument("--no-m3-colbert", action="store_true", help="bge_m3 의 ColBERT 재채점을 끈다")
    ap.add_argument("--m3-ablation", action="store_true", help="bge_m3 를 dense / sparse / dense+sparse / 전부 4가지로 나눠 비교")
    ap.add_argument("--m3-cache", default=None, help="dense/sparse 인코딩 캐시(SQLite). 재실행 시 새 텍스트만 인코딩")
    ap.add_argument("--reranker-model", default=None, help="예: Dongjin-kr/ko-reranker (선택, 모든 후보에 리랭커 변형을 추가)")
    ap.add_argument("--top-k", type=int, default=5); ap.add_argument("--strategy", default="mean", choices=["mean", "rank", "normalize_mean"])
    ap.add_argument("--overlap-threshold", type=float, default=0.6)
    ap.add_argument("--max-file-mb", type=int, default=60); ap.add_argument("--parse-cache", default=None)
    a = ap.parse_args(argv)

    root, out = Path(a.root).resolve(), Path(a.out)
    loader = SafeDocumentLoader(root, EXTS, a.max_file_mb * 1024 * 1024)
    if a.parse_cache:
        from intellectagent.parse_cache import CachingDocumentLoader, ParseCache
        loader = CachingDocumentLoader(loader, ParseCache(Path(a.parse_cache)))
    docs, failed = [], []
    for p in loader.discover():
        try:
            docs.append(loader.load(p))
        except Exception as exc:
            failed.append({"path": str(p), "error": f"{type(exc).__name__}: {exc}"})
    if not docs:
        print("문서를 하나도 읽지 못했습니다.", failed, file=sys.stderr)
        return 2

    approved = load_approved(Path(a.qa))
    answerable = [r for r in approved if not r.should_fallback]
    skipped_unanswerable = len(approved) - len(answerable)
    if not answerable:
        print("approved 상태의 답 있는 질문이 없습니다. qa_draft.jsonl 을 검수해 status 를 approved 로 바꾸세요.", file=sys.stderr)
        return 2
    known = {d.doc_id for d in docs}
    orphan = sorted({i for r in answerable for i in r.relevant_doc_ids} - known)
    if orphan:
        print(f"질문의 정답 문서 ID 가 --root 의 문서와 맞지 않습니다({len(orphan)}개): {orphan[:5]}. "
              "gen_qa.py 때와 같은 폴더·같은 파일인지 확인하세요.", file=sys.stderr)
        return 2
    empty_gt = [r.case_id for r in answerable if not r.gt_contents]
    if empty_gt:
        print(f"정답 근거 텍스트(gt_contents)가 없는 질문: {empty_gt[:5]}", file=sys.stderr)
        return 2

    embedder = HashEmbedder() if a.embedder == "hash" else SentenceTransformerEmbedder(a.embedder)
    rerankers, specs = {}, parse_specs(a.retrievers, a.cc_weight, a.m3_weights, not a.no_m3_colbert, a.m3_ablation)
    m3_encoder = m3_cache = None
    if any(s.kind == "bge_m3" for s in specs):
        from intellectagent.bge_m3 import FlagM3Encoder, HashM3Encoder, M3Cache
        m3_encoder = HashM3Encoder() if a.embedder == "hash" else FlagM3Encoder(a.embedder)
        m3_cache = M3Cache(Path(a.m3_cache)) if a.m3_cache else None
    if a.reranker_model:
        from intellectagent.retrieval import CrossEncoderReranker
        rerankers["ko"] = CrossEncoderReranker(a.reranker_model)
        specs += [RetrieverSpec(s.kind, s.tokenizer, s.idf_variant, s.rrf_k, s.cc_weight, s.cc_normalize, s.candidate_k, "ko", s.m3_weights, s.m3_colbert)
                  for s in list(specs) if s.kind != "bge_m3"]
    cands = build_candidates(docs, parse_chunks(a.chunks), specs, embedder=embedder, rerankers=rerankers, m3_encoder=m3_encoder, m3_cache=m3_cache)

    full = run_grid(cands, answerable, top_k=a.top_k, metrics=METRICS, strategy=a.strategy, out_dir=out)
    hi = [r for r in answerable if r.lexical_overlap >= a.overlap_threshold]
    lo = [r for r in answerable if r.lexical_overlap < a.overlap_threshold]
    parts = {}
    for name, subset in (("overlap_high", hi), ("overlap_low", lo)):
        if subset:
            parts[name] = run_grid(cands, subset, top_k=a.top_k, metrics=METRICS, strategy=a.strategy, out_dir=out / name)

    warn = []
    if len(answerable) < MIN_RELIABLE:
        warn.append(f"질문이 {len(answerable)}개뿐입니다(<{MIN_RELIABLE}). 순위 차이는 우연일 수 있습니다.")
    if a.embedder == "hash":
        warn.append("HashEmbedder 를 썼습니다. vector/hybrid/bge_m3 결과는 의미 검색 성능이 아닙니다(배관 확인용).")
    if failed:
        warn.append(f"읽지 못한 문서 {len(failed)}개: " + ", ".join(Path(f['path']).name for f in failed))
    md = [f"# 청킹 x 검색 방식 비교 (실문서)\n", f"- 문서 {len(docs)}개, 질문 {len(answerable)}개(답 없는 질문 {skipped_unanswerable}개 제외), top_k={a.top_k}, "
          f"임베더={a.embedder}, 선택 전략={a.strategy}", f"- 청킹: {a.chunks}", "- ★ = tok_f1 최고 후보. **청킹 간 비교는 tok_* 열을 본다.** (grid_result.json 의 best 는 --strategy 가 지표 평균으로 고른 것이라 다를 수 있다)\n"]
    md += [f"> ⚠ {w}" for w in warn] + [""]
    md.append(table(f"전체 ({len(answerable)}문항)", full))
    for name, res in parts.items():
        md.append(table(f"{name} (lexical_overlap {'≥' if name.endswith('high') else '<'} {a.overlap_threshold}, {len(hi) if name.endswith('high') else len(lo)}문항)", res))
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    (out / "run_info.json").write_text(json.dumps({"args": vars(a), "documents": len(docs), "failed": failed, "questions": len(answerable),
                                                  "skipped_unanswerable": skipped_unanswerable, "warnings": warn, "best_by_tok_f1": star(full), "best_by_strategy": full["best"]},
                                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(md))
    print(f"\n저장: {out}/report.md, grid_result.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intellectagent.evaluation import load_cases
from intellectagent.loaders import SafeDocumentLoader
from intellectagent.reranking import TokenOverlapReranker
from intellectagent.retrieval import CrossEncoderReranker, NoopReranker
from intellectagent.retrieval_benchmark import RetrievalBenchmarkConfig, run_retrieval_benchmark


def build_reranker(name: str, model: str):
    if name == "none":
        return NoopReranker
    if name == "token-overlap":
        return TokenOverlapReranker
    if name == "bge":
        return lambda: CrossEncoderReranker(model)
    raise ValueError(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/sample_docs")
    ap.add_argument("--eval", default="data/eval/week2/eval15.jsonl")
    ap.add_argument("--reranker", choices=["none", "token-overlap", "bge"], default="none")
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--candidate-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    loader = SafeDocumentLoader(root, (".md", ".txt", ".pdf"), 10 * 1024 * 1024)
    docs = [loader.load(p) for p in loader.discover()]
    cases = load_cases(Path(args.eval))
    cfg = RetrievalBenchmarkConfig(
        experiment_name=f"week3_{args.reranker}_c{args.candidate_k}_k{args.top_k}",
        seed=args.seed,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        reranker_name=args.reranker if args.reranker != "bge" else args.model,
    )
    result = run_retrieval_benchmark(
        docs,
        cases,
        cfg,
        reranker_factory=build_reranker(args.reranker, args.model),
    )
    out = Path(args.out or f"evidence/week3/{args.reranker}_c{args.candidate_k}_k{args.top_k}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

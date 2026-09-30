from __future__ import annotations
import argparse
import json
from pathlib import Path

from intellectagent.chunk_strategies import ChunkStrategyConfig
from intellectagent.evaluation import load_cases
from intellectagent.experiment import RetrievalExperimentConfig, run_retrieval_experiment, save_experiment_config
from intellectagent.loaders import SafeDocumentLoader


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/sample_docs")
    ap.add_argument("--eval", default="data/eval/week2/eval15.jsonl")
    ap.add_argument("--strategy", choices=["semantic", "length"], required=True)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--size", type=int, default=700)
    ap.add_argument("--overlap", type=int, default=80)
    ap.add_argument("--out-dir", default="evidence/week2")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    loader = SafeDocumentLoader(root, (".md", ".txt", ".pdf"), 10 * 1024 * 1024)
    docs = [loader.load(p) for p in loader.discover()]
    cases = load_cases(Path(args.eval))
    strategy = ChunkStrategyConfig(name=args.strategy, size=args.size, overlap=args.overlap)
    cfg = RetrievalExperimentConfig(
        experiment_name=f"week2_{args.strategy}",
        seed=args.seed,
        top_k=args.top_k,
        strategy=strategy,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_experiment_config(out_dir / f"{args.strategy}_config.json", cfg)
    result = run_retrieval_experiment(docs, cases, cfg)
    out = out_dir / f"{args.strategy}_result.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"saved -> {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

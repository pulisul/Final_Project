#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-src}"
python scripts/week3/run_retrieval_benchmark.py --reranker none --candidate-k 10 --top-k 5 --out evidence/week3/none_c10_k5.json
python scripts/week3/run_retrieval_benchmark.py --reranker token-overlap --candidate-k 10 --top-k 5 --out evidence/week3/token-overlap_c10_k5.json
python scripts/week3/compare_retrieval.py
python scripts/week3/tune_candidate_k.py
python -m pytest -q
python scripts/week3/run_week3_gate.py

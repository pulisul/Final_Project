#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/week6/build_eval20.py
rm -f evidence/week6/eval_metadata.sqlite3 evidence/week6/eval_checkpoint.json
python scripts/week6/run_deterministic_eval.py --system baseline --seed 20260921
rm -f evidence/week6/eval_metadata.sqlite3 evidence/week6/eval_checkpoint.json
python scripts/week6/run_deterministic_eval.py --system reranked --seed 20260921
python scripts/week6/compare_eval_runs.py
python scripts/week6/make_human_eval_template.py
python scripts/week6/run_week6_gate.py

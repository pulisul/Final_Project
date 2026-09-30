#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
python scripts/week2/build_eval15.py
python scripts/week2/run_chunk_experiment.py --strategy semantic
python scripts/week2/run_chunk_experiment.py --strategy length
python scripts/week2/compare_chunk_strategies.py
python scripts/week2/run_week2_gate.py

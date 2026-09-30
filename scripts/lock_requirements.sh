#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip freeze --all | sort > requirements-lock.txt
python -m pip check
python -m pytest -q
python scripts/run_release_gate.py --mode strict

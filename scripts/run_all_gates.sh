#!/usr/bin/env bash
set -u
export PYTHONPATH=src
python scripts/run_release_gate.py --mode smoke
SMOKE=$?
python scripts/run_release_gate.py --mode strict
STRICT=$?
echo "smoke_exit=$SMOKE strict_exit=$STRICT"
exit $STRICT

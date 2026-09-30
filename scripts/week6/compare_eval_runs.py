#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
A=json.loads((ROOT/'evidence/week6/baseline_summary.json').read_text(encoding='utf-8'))
B=json.loads((ROOT/'evidence/week6/reranked_summary.json').read_text(encoding='utf-8'))
from sys import path as syspath
syspath.insert(0,str(ROOT/'src'))
from intellectagent.evaluation_week6 import compare_week6_runs
out=compare_week6_runs(A,B)
(ROOT/'evidence/week6/evaluation_comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))

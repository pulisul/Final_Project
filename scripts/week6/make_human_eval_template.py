#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from intellectagent.human_eval import write_human_eval_template
write_human_eval_template(ROOT/'evidence/week6/reranked_raw.jsonl',ROOT/'evidence/week6/human_eval.csv')
print(ROOT/'evidence/week6/human_eval.csv')

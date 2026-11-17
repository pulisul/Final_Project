#!/usr/bin/env python3
import argparse,json,sys
from intellectagent.promotion import PromotionGate
p=argparse.ArgumentParser(); p.add_argument("feature"); p.add_argument("candidate_json"); a=p.parse_args()
c=json.load(open(a.candidate_json,encoding="utf-8")); r=PromotionGate().evaluate(a.feature,c)
print(json.dumps({"feature":r.feature,"passed":r.passed,"reasons":r.reasons},ensure_ascii=False,indent=2))
sys.exit(0 if r.passed else 2)

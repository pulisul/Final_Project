"""AutoRAG legacy strategy.py 의 select_best 와 우리 grid_search.select_best 를 동점 포함 무작위로 대조한다.
사용: python scripts/autorag/crosscheck_strategy.py <AutoRAG>/legacy/autorag/strategy.py   (pandas, numpy 필요)
"""
import sys, random, importlib.util
from fractions import Fraction
import pandas as pd, numpy as np
sys.path.insert(0, "src")
spec = importlib.util.spec_from_file_location("strat", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from intellectagent.grid_search import select_best
cols = ["a", "b", "c"]
def _exact_tie(rows, cols, x, y):
    def tot(name):
        t = Fraction(0)
        for c in cols:
            vals = [Fraction(r[c]) for r in rows]; lo, hi = min(vals), max(vals)
            if hi != lo: t += (Fraction(next(r for r in rows if r["candidate"] == name)[c]) - lo) / (hi - lo)
        return t
    return tot(x) == tot(y)
rng = random.Random(1); diff = tot = noise = 0; ex = None
for _ in range(2000):
    n = rng.randint(2, 5)
    rows = [{"candidate": f"c{i}", **{c: rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]) for c in cols}} for i in range(n)]
    dfs = [pd.DataFrame({c: [r[c]] for c in cols}) for r in rows]
    for strat, orig in (("mean", "mean"), ("rank", "rank"), ("normalize_mean", "normalize_mean")):
        _, i = m.select_best(dfs, cols, list(range(n)), orig)
        mine = select_best(rows, cols, strat)["candidate"]
        tot += 1
        if mine != f"c{i}":
            if strat == "normalize_mean" and _exact_tie(rows, cols, mine, f"c{i}"):
                noise += 1  # 수학적으로 동점: 원본은 부동소수 잡음으로, 우리는 12자리 반올림 후 첫 후보로 결정
                continue
            diff += 1
            if ex is None: ex = (strat, rows, mine, f"c{i}")
print("cases", tot, "mismatch", diff, "exact_tie_float_noise", noise); print(ex)

sys.exit(1 if diff else 0)

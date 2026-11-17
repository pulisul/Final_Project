"""원본 AutoRAG 와 대조 (2): 내용 기반 토큰 지표, hybrid_cc.

사용: PYTHONPATH=src python scripts/autorag/crosscheck_more.py <AutoRAG-main/legacy/autorag 경로>   (numpy, pandas 필요)
원본 소스에서 함수만 AST 로 추출해 실행한다(무거운 의존성을 import 하지 않기 위해).
"""
from __future__ import annotations

import ast
import itertools
import random
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

from intellectagent.hybrid import cc_fuse
from intellectagent.models import SearchHit
from intellectagent import retrieval_contents as ours


def extract(path: Path, names: set[str], ns: dict) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            n.decorator_list = []  # @autorag_metric 등 제거: 원본 본문 로직만 쓴다
            body.append(n)
        elif isinstance(n, ast.Assign) and any(getattr(t, "id", None) in names for t in n.targets):
            body.append(n)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
    return ns


WORDS = ["서버", "백업", "휴가", "신청", "The", "a", "an", "API", "bm25", "문서,", "보관!", "30일", "승인.", "팀장", "Backup"]


def rand_text(rng, lo=0, hi=10):
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(lo, hi)))




def main():
    A = Path(sys.argv[1])
    import numpy as np, pandas as pd
    # --- 토큰 지표 ---
    ns = {"re": __import__("re"), "string": __import__("string"), "Counter": Counter, "itertools": itertools, "np": np}
    extract(A / "utils/util.py", {"normalize_string"}, ns)
    extract(A / "evaluation/metric/retrieval_contents.py", {"single_token_f1"}, ns)
    ref_stf, ref_norm = ns["single_token_f1"], ns["normalize_string"]

    def ref_metric(idx, pred, gt):  # 원본 retrieval_token_* 본문과 동일 (metric_input 래퍼만 제거)
        gt = itertools.chain.from_iterable(gt)
        res = list(map(lambda x: ref_stf(x[1], x[0]), list(itertools.product(pred, gt))))
        col = list(zip(*res))[idx]
        return float(np.array(list(col)).reshape(len(pred), -1).max(axis=1).mean())

    worst = 0.0; n = 0; norm_bad = 0
    for _ in range(3000):
        pred = [rand_text(rng, 1, 8) for _ in range(rng.randint(1, 4))]
        gt = [[rand_text(rng, 1, 8) for _ in range(rng.randint(1, 2))] for _ in range(rng.randint(1, 3))]
        for idx, fn in enumerate((ours.retrieval_token_precision, ours.retrieval_token_recall, ours.retrieval_token_f1)):
            worst = max(worst, abs(fn(gt, pred) - ref_metric(idx, pred, gt))); n += 1
        s = rand_text(rng, 0, 10)
        norm_bad += ours.normalize_string(s) != ref_norm(s)
    print(f"[토큰지표] {n}건 최대 절대오차 {worst:.2e}, normalize_string 불일치 {norm_bad}건")

    # --- hybrid_cc ---
    ns2 = {"np": np, "pd": pd, "List": List, "Tuple": Tuple}
    extract(A / "nodes/hybridretrieval/hybrid_cc.py",
            {"normalize_mm", "normalize_tmm", "normalize_z", "normalize_dbsf", "normalize_method_dict", "fuse_per_query"}, ns2)
    bad_score = bad_ids = ties = tot = 0
    for _ in range(3000):
        pool = [f"c{i}" for i in range(14)]
        s_ids = rng.sample(pool, rng.randint(2, 9)); l_ids = rng.sample(pool, rng.randint(2, 9))
        s_sc = sorted((rng.uniform(-1, 1) for _ in s_ids), reverse=True)
        l_sc = sorted((rng.uniform(0, 12) for _ in l_ids), reverse=True)
        method = rng.choice(["mm", "tmm", "z", "dbsf"]); w = rng.choice([0.0, 0.3, 0.5, 0.7, 1.0]); k = rng.choice([3, 5, 20])
        r_ids, r_sc = ns2["fuse_per_query"](s_ids, l_ids, s_sc, l_sc, normalize_method=method, weight=w, top_k=k,
                                            semantic_theoretical_min_value=-1.0, lexical_theoretical_min_value=0.0)
        mine = cc_fuse([SearchHit(i, "d", "t", s) for i, s in zip(s_ids, s_sc)],
                       [SearchHit(i, "d", "t", s) for i, s in zip(l_ids, l_sc)], top_k=k, weight=w, normalize_method=method)
        tot += 1
        if len(mine) != len(r_sc) or any(abs(a.score - b) > 1e-9 for a, b in zip(mine, r_sc)):
            bad_score += 1; continue
        if [h.chunk_id for h in mine] != list(r_ids):
            # 동점이면 pandas(비안정 정렬)와 순서/경계 선택만 다를 수 있다. top_k 경계 밖의 동점도 보려고 전체 목록으로 다시 판정한다.
            _, full = ns2["fuse_per_query"](s_ids, l_ids, s_sc, l_sc, normalize_method=method, weight=w, top_k=100,
                                            semantic_theoretical_min_value=-1.0, lexical_theoretical_min_value=0.0)
            if len({round(x, 9) for x in full}) < len(full): ties += 1
            else: bad_ids += 1
    print(f"[hybrid_cc] {tot}건: 점수열 불일치 {bad_score}, (동점 아닌데) id 순서 불일치 {bad_ids}, 동점으로 순서만 다름 {ties}")
    # 퇴화 입력: 원본은 NaN -> 0 처리
    for method in ("mm", "tmm", "z", "dbsf"):
        r_ids, r_sc = ns2["fuse_per_query"](["a", "b"], ["b", "c"], [0.5, 0.5], [3.0, 3.0], normalize_method=method, weight=0.5,
                                            top_k=3, semantic_theoretical_min_value=-1.0, lexical_theoretical_min_value=0.0)
        mine = cc_fuse([SearchHit("a", "d", "t", .5), SearchHit("b", "d", "t", .5)], [SearchHit("b", "d", "t", 3.0), SearchHit("c", "d", "t", 3.0)],
                       top_k=3, weight=0.5, normalize_method=method)
        ref = [(i, round(0.0 if s != s else s, 9)) for i, s in zip(r_ids, r_sc)]
        print(f"   퇴화입력 {method:4s} 원본={ref} 우리={[(h.chunk_id, round(h.score, 9)) for h in mine]}")
    return 1 if (worst > 1e-9 or norm_bad or bad_score or bad_ids) else 0


rng = random.Random(20261019)
if __name__ == "__main__":
    sys.exit(main())

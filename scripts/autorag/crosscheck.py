"""원본 AutoRAG(legacy)와 우리 구현이 같은 결과를 내는지 직접 대조한다.

사용: PYTHONPATH=src python scripts/autorag/crosscheck.py <AutoRAG-main/legacy/autorag 경로> [--n 3000]
- 지표 6종: 원본 함수와 무작위 입력 n 건 비교
- BM25: rank_bm25.BM25Okapi 와 점수 비교 (pip install rank-bm25)
- RRF: 원본 rrf_pure 소스를 그대로 추출해 비교 (pandas 필요)
원본 패키지 전체를 import 하지 않는다(무거운 의존성 회피): 필요한 모듈만 경로로 불러온다.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import random
import sys
import types
from pathlib import Path

from intellectagent import retrieval_metrics as ours
from intellectagent.hybrid import rrf_fuse
from intellectagent.lexical import BM25Index, tok_char_bigram, tok_word
from intellectagent.models import Chunk, SearchHit


def load_autorag_metrics(autorag_dir: str):
    def fake(name, path):
        m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m
    fake("autorag", autorag_dir); fake("autorag.evaluation", autorag_dir + "/evaluation")
    fake("autorag.evaluation.metric", autorag_dir + "/evaluation/metric"); fake("autorag.schema", autorag_dir + "/schema")
    return importlib.import_module("autorag.evaluation.metric.retrieval"), importlib.import_module("autorag.schema.metricinput").MetricInput


def check_metrics(ref, MetricInput, n, rng):
    ids = [f"d{i}" for i in range(12)]
    worst = {m: 0.0 for m in ours.METRICS}
    skipped = {m: 0 for m in ours.METRICS}  # 원본이 None(계산 제외)을 돌려준 횟수
    bad_empty = 0
    for _ in range(n):
        groups = [rng.sample(ids, rng.randint(1, 3)) for _ in range(rng.randint(1, 4))]
        pred = rng.sample(ids, rng.randint(0, 8))
        for name, fn in ours.METRICS.items():
            a = fn(groups, pred)
            b = getattr(ref, name)([MetricInput(retrieval_gt=groups, retrieved_ids=pred)])[0]
            if b is None:  # 원본은 검색 결과가 비면 점수를 None 으로 둔다(우리는 0점으로 센다 — 아래에서 확인)
                skipped[name] += 1
                bad_empty += 0 if (not pred and a == 0.0) else 1
                continue
            worst[name] = max(worst[name], abs(a - b))
    return worst, skipped, bad_empty


def check_bm25(rng):
    from rank_bm25 import BM25Okapi
    vocab = [f"w{i}" for i in range(40)]
    worst = 0.0
    for _ in range(200):
        docs = [" ".join(rng.choices(vocab, k=rng.randint(3, 30))) for _ in range(rng.randint(2, 25))]
        chunks = [Chunk(f"c{i}", f"d{i}", t, "x", i) for i, t in enumerate(docs)]
        mine = BM25Index(chunks, tok_word)
        ref = BM25Okapi([tok_word(t) for t in docs])
        q = " ".join(rng.choices(vocab, k=rng.randint(1, 5)))
        a = mine.scores(q); b = ref.get_scores(tok_word(q))
        worst = max(worst, max(abs(x - y) for x, y in zip(a, b)))
    return worst


def check_rrf(autorag_dir, rng):
    import pandas as pd
    src = Path(autorag_dir, "nodes/hybridretrieval/hybrid_rrf.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    wanted = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("rrf_pure", "rrf_calculate")]
    ns = {"pd": pd, "Tuple": tuple, "List": list}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "rrf_ref", "exec"), ns)
    mism = 0
    for _ in range(500):
        pool = [f"c{i}" for i in range(15)]
        lists, id_lists, score_lists = [], [], []
        for _ in range(rng.randint(2, 3)):
            ids = rng.sample(pool, rng.randint(1, 10))
            scores = sorted((round(rng.random(), 1) for _ in ids), reverse=True)  # 일부러 동점이 나오게
            id_lists.append(ids); score_lists.append(scores)
            lists.append([SearchHit(i, "d", "t", s) for i, s in zip(ids, scores)])
        k = rng.choice([10, 60])
        ref_ids, ref_sc = ns["rrf_pure"](tuple(id_lists), tuple(score_lists), k, 5)
        mine = rrf_fuse(lists, top_k=5, k=k)
        # 동점 처리 순서만 다를 수 있으므로 (id, 점수) 집합이 아니라 '점수열'과 '점수별 id 집합'을 비교
        if [round(h.score, 12) for h in mine] != [round(s, 12) for s in ref_sc]:
            mism += 1
    return mism


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("autorag_dir"); ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args(); rng = random.Random(20261019)
    ref, MI = load_autorag_metrics(a.autorag_dir)
    w, sk, bad = check_metrics(ref, MI, a.n, rng)
    print(f"[지표] {a.n}건 무작위 대조 (원본이 None 을 준 건은 제외하고 비교)")
    for k, v in w.items(): print(f"   {k:20s} 최대 절대오차 {v:.2e}   원본 None {sk[k]}건")
    print(f"   원본 None 인 모든 경우가 '검색 결과 비어 있음' 이고 우리 값이 0.0 인가? 위반 {bad}건")
    print(f"[BM25] rank_bm25 와 점수 최대 절대오차(200 코퍼스): {check_bm25(rng):.2e}")
    print(f"[RRF ] 원본 rrf_pure 와 점수열 불일치 건수(500 케이스, 동점 포함): {check_rrf(a.autorag_dir, rng)}")


if __name__ == "__main__":
    main()

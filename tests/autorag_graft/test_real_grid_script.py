import importlib.util
import json
import sys
from pathlib import Path

import pytest

from intellectagent.loaders import SafeDocumentLoader
from intellectagent.qa_gen import QARecord, write_jsonl

SPEC = importlib.util.spec_from_file_location("rrg", Path(__file__).parents[2] / "scripts/autorag/run_real_grid.py")
rrg = importlib.util.module_from_spec(SPEC); sys.modules["rrg"] = rrg; SPEC.loader.exec_module(rrg)

T1 = "서버 백업 지침: 매일 새벽 2시에 전체 백업을 수행하고 30일간 보관한다. 복구 훈련은 분기마다 실시한다. " * 4
T2 = "원격 근무 VPN 접속 절차: 사내 인증서를 설치한 뒤 이중 인증을 통과해야 접속할 수 있다. " * 4


def setup(tmp_path, status="approved", bad_doc_id=False):
    root = tmp_path / "docs"; root.mkdir()
    (root / "a.md").write_text(T1, encoding="utf-8"); (root / "b.md").write_text(T2, encoding="utf-8")
    ids = {d.metadata["name"]: d.doc_id for d in map(SafeDocumentLoader(root, (".md",), 10**6).load, sorted(root.glob("*.md")))}
    qa = [QARecord("q1", "서버 백업은 언제 수행하나요", ("zzz" if bad_doc_id else ids["a.md"],), gt_contents=("매일 새벽 2시에 전체 백업을 수행하고 30일간 보관한다",), status=status),
          QARecord("q2", "VPN 접속에 필요한 인증은 무엇인가요", (ids["b.md"],), gt_contents=("사내 인증서를 설치한 뒤 이중 인증을 통과해야 접속할 수 있다",), status=status),
          QARecord("q3", "우주선 엔진은?", (), should_fallback=True, status=status)]
    write_jsonl(qa, tmp_path / "qa.jsonl")
    return root


def run(tmp_path, root, *extra):
    return rrg.main(["--root", str(root), "--qa", str(tmp_path / "qa.jsonl"), "--out", str(tmp_path / "out"), "--embedder", "hash",
                     "--chunks", "100:10,200:20", "--retrievers", "bm25,hybrid_rrf", "--top-k", "2", *extra])


def test_runs_end_to_end_and_writes_report(tmp_path, capsys):
    assert run(tmp_path, setup(tmp_path)) == 0
    rep = (tmp_path / "out/report.md").read_text(encoding="utf-8")
    assert "질문 2개(답 없는 질문 1개 제외)" in rep and "★" in rep and "HashEmbedder" in rep and "<30" in rep
    info = json.loads((tmp_path / "out/run_info.json").read_text(encoding="utf-8"))
    assert info["questions"] == 2 and info["skipped_unanswerable"] == 1
    rows = (tmp_path / "out/grid_result.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + 2 * 2   # 청킹 2 x 검색 2


def test_draft_only_questions_are_refused(tmp_path, capsys):
    assert run(tmp_path, setup(tmp_path, status="draft")) == 2
    assert "approved" in capsys.readouterr().err


def test_mismatched_documents_are_refused(tmp_path, capsys):
    assert run(tmp_path, setup(tmp_path, bad_doc_id=True)) == 2
    assert "맞지 않습니다" in capsys.readouterr().err


def test_star_is_best_token_f1_not_the_mean_strategy_pick():
    res = {"best": "x", "rows": [{"candidate": "x", "retrieval_token_f1": 0.1}, {"candidate": "y", "retrieval_token_f1": 0.3}]}
    assert rrg.star(res) == "y"


def test_context_header_and_bge_m3_ablation_options(tmp_path):
    root = setup(tmp_path)
    assert run_with(tmp_path, root, "--chunks", "100:10,100:10:ctx", "--retrievers", "bm25,bge_m3", "--m3-ablation") == 0
    rows = (tmp_path / "out/grid_result.csv").read_text(encoding="utf-8")
    for name in ("length-100-10|", "length-100-10+ctx|", "bge_m3+d-w1:0:0", "bge_m3+s-w0:1:0", "bge_m3+ds-w2:1:0", "bge_m3+dsc"):
        assert name in rows
    assert len(rows.splitlines()) == 1 + 2 * (1 + 4)       # 청킹 2 x (bm25 + m3 4종)


def run_with(tmp_path, root, *extra):
    return rrg.main(["--root", str(root), "--qa", str(tmp_path / "qa.jsonl"), "--out", str(tmp_path / "out"), "--embedder", "hash", "--top-k", "2", *extra])


def test_parse_chunks_accepts_ctx_suffix():
    c = rrg.parse_chunks("300:30,300:30:ctx,500")
    assert [(x.size, x.overlap, x.context_header) for x in c] == [(300, 30, False), (300, 30, True), (500, 0, False)]

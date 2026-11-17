"""새 검색기가 build_components -> CoreRAG 실경로에서 동작하는지 (HashEmbedder, 외부 모델 없음)."""
from pathlib import Path

import pytest

from intellectagent.config import CoreConfig
from intellectagent.factory import build_components
from intellectagent.hybrid import GatedHybridRetriever, StoreBM25
from intellectagent.indexing import Indexer
from intellectagent.loaders import SafeDocumentLoader
from intellectagent.metadata import MetadataDB
from intellectagent.models import Chunk
from intellectagent.pipeline import CoreRAG
from intellectagent.retrieval import Retriever
from intellectagent.vectorstore import InMemoryVectorStore

DOCS = {
    "backup.md": "서버 백업 지침: 매일 새벽 2시에 전체 백업을 수행하고 30일간 보관한다. 복구 훈련은 분기마다 실시한다. " * 3,
    "vpn.md": "원격 근무 VPN 접속 절차: 사내 인증서를 설치한 뒤 이중 인증을 통과해야 접속할 수 있다. " * 3,
    "herb.md": "황기와 당귀는 배수가 잘되는 토양에서 재배하며 파종 시기는 봄이다. " * 3,
}


def make(tmp_path: Path, mode: str, **flags):
    root = tmp_path / "docs"; root.mkdir(exist_ok=True)
    for n, t in DOCS.items():
        (root / n).write_text(t, encoding="utf-8")
    cfg = CoreConfig(project_root=tmp_path, allowed_root=root, index_dir=tmp_path / "idx", metadata_db=tmp_path / "m.db",
                     evidence_dir=tmp_path / "ev", allowed_extensions=(".md", ".txt"), semantic_min_chars=20, semantic_max_chars=200,
                     feature_flags={"retrieval_mode": mode, **flags})
    emb, store, retr, gen, pipe = build_components(cfg)
    Indexer(SafeDocumentLoader(root, cfg.allowed_extensions, cfg.max_file_bytes), emb, store, MetadataDB(cfg.metadata_db), cfg).index_all(tmp_path / "ck.json")
    return cfg, store, retr, pipe


def test_default_mode_is_unchanged_vector_retriever(tmp_path):
    _, _, retr, pipe = make(tmp_path, "vector")
    assert type(retr) is Retriever and isinstance(pipe, CoreRAG)


@pytest.mark.parametrize("mode", ["bm25", "hybrid_rrf", "hybrid_cc"])
def test_every_mode_answers_with_the_right_document(tmp_path, mode):
    _, _, _, pipe = make(tmp_path, mode)
    a = pipe.ask("원격 근무 VPN 접속하려면 어떻게 하나요")
    assert not a.fallback and a.hits and "VPN" in a.hits[0].text


def test_rrf_scores_are_below_the_gate_yet_answer_is_not_rejected(tmp_path):
    cfg, _, _, pipe = make(tmp_path, "hybrid_rrf")
    a = pipe.ask("서버 백업은 언제 수행하나요")
    assert max(h.score for h in a.hits) < cfg.fallback_min_score       # 융합 점수 척도는 게이트보다 훨씬 낮다
    assert "백업" in a.hits[0].text and not a.fallback


def test_unrelated_question_still_falls_back_in_every_mode(tmp_path):
    for mode in ("bm25", "hybrid_rrf", "hybrid_cc"):
        (tmp_path / mode).mkdir()
        _, _, _, pipe = make(tmp_path / mode, mode)
        a = pipe.ask("zxqv 우주선 엔진 qqqq")
        assert a.fallback, mode


def test_bm25_picks_up_documents_added_after_construction(tmp_path):
    store = InMemoryVectorStore()
    bm = StoreBM25(store)
    assert bm.search("백업", 3) == []
    store.upsert([Chunk("c1", "d1", "서버 백업 지침", "length", 0)], [[1.0, 0.0]])
    assert [h.chunk_id for h in bm.search("백업", 3)] == ["c1"]
    store.upsert([Chunk("c2", "d2", "백업 백업 백업 복구 절차", "length", 0)], [[0.0, 1.0]])
    assert {h.chunk_id for h in bm.search("백업", 3)} == {"c1", "c2"}
    store.delete_doc("d1")
    assert [h.chunk_id for h in bm.search("백업", 3)] == ["c2"]


def test_bad_mode_and_tokenizer_fail_loudly(tmp_path):
    with pytest.raises(ValueError):
        make(tmp_path, "nope")
    (tmp_path / "x").mkdir()
    with pytest.raises(ValueError):
        make(tmp_path / "x", "bm25", bm25_tokenizer="nope")


def test_hybrid_gate_score_metadata_only_in_hybrid_hits(tmp_path):
    _, _, retr, _ = make(tmp_path, "hybrid_rrf")
    assert isinstance(retr, GatedHybridRetriever)
    hits = retr.search("서버 백업", 3)
    assert hits and "gate_score" in hits[0].metadata


def test_store_bm25_follows_a_persistent_chroma_store(tmp_path):
    pytest.importorskip("chromadb")
    from intellectagent.vectorstore import ChromaVectorStore

    st = ChromaVectorStore(str(tmp_path / "chroma"), "tcoll")
    st.upsert([Chunk("c1", "d1", "서버 백업 지침", "length", 0, {"source": "a.md"})], [[1.0, 0.0]])
    bm = StoreBM25(st)
    h = bm.search("백업", 3)
    assert [(x.chunk_id, x.doc_id, x.metadata.get("source")) for x in h] == [("c1", "d1", "a.md")]
    st.upsert([Chunk("c2", "d2", "백업 복구 절차 백업", "length", 1)], [[0.0, 1.0]])
    assert len(bm.search("백업", 3)) == 2
    st.delete_doc("d1")
    assert [x.chunk_id for x in bm.search("백업", 3)] == ["c2"]


def test_bge_m3_mode_and_context_headers_work_through_the_real_pipeline(tmp_path):
    from intellectagent.bge_m3 import M3Retriever

    cfg, store, retr, pipe = make(tmp_path, "bge_m3", context_headers=True)
    assert isinstance(retr, M3Retriever) and retr.warm() == len(store.rows)
    # 머리말이 색인에 들어갔는지(원문은 그대로): 모든 청크에 [파일명] 이 달려 있다
    assert all(c.metadata.get("context", "").startswith("[") for c, _ in store.rows.values())
    assert all(not c.text.startswith("[") for c, _ in store.rows.values())
    a = pipe.ask("원격 근무 VPN 접속 절차는 무엇인가요")
    assert not a.fallback and "VPN" in a.hits[0].text and "gate_score" in a.hits[0].metadata
    assert pipe.ask("zxqv 우주선 엔진 qqqq").fallback


def test_cli_index_warms_bge_m3(tmp_path, monkeypatch, capsys):
    import json
    from intellectagent import cli

    root = tmp_path / "docs"; root.mkdir(); (root / "a.md").write_text("서버 백업 지침: 매일 새벽 2시에 전체 백업을 수행한다. " * 4, encoding="utf-8")
    (tmp_path / "config").mkdir()
    (tmp_path / "config/core.json").write_text(json.dumps({
        "allowed_root": "docs", "index_dir": "idx", "metadata_db": "m.db", "evidence_dir": "ev", "allowed_extensions": [".md"],
        "semantic_min_chars": 20, "semantic_max_chars": 200, "feature_flags": {"retrieval_mode": "bge_m3"}}), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["intellectagent", "--config", str(tmp_path / "config/core.json"), "index"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["m3_chunks_encoded"] >= 1

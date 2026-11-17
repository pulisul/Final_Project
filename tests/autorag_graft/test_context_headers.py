from pathlib import Path

import pytest

from intellectagent.chunk_grid import RetrieverSpec, build_candidates, chunk_label
from intellectagent.chunk_strategies import ChunkStrategyConfig, chunk_with_strategy
from intellectagent.context_headers import add_context_headers, find_headings
from intellectagent.hashing import sha256_text
from intellectagent.lexical import BM25Index, tok_char_bigram
from intellectagent.models import Chunk, Document, index_text

MD = """# 서버 운영 지침

## 3장 백업

### 3.2 보관 기간

전체 백업은 30일간 보관한다. 이 기간이 지나면 자동으로 삭제된다.

## 4장 보안

### 4.1 접속 절차

원격 근무자는 인증서를 설치해야 한다. 그 뒤에 이중 인증을 통과한다.
"""


def doc(text=MD, name="ops.md"):
    return Document(sha256_text(text)[:16], Path(name), text, sha256_text(text), {"name": name})


def test_heading_detection_is_conservative():
    text = "제1장 총칙\n1. 서버 백업을 매일 수행한다.\n2.1 접속 절차\n3. 보안\n그냥 문장입니다.\n[슬라이드 4]\n[시트 예산]\n# 제목\n"
    assert [(d, t) for _, d, t in find_headings(text)] == [(1, "제1장 총칙"), (2, "2.1 접속 절차"), (1, "3. 보안"), (1, "슬라이드 4"), (1, "시트 예산"), (1, "제목")]


def test_each_chunk_gets_its_section_path_and_text_is_unchanged():
    d = doc()
    base = chunk_with_strategy(d, ChunkStrategyConfig("semantic", min_chars=1, max_chars=40))
    ctx = add_context_headers(d, base)
    assert [c.text for c in ctx] == [c.text for c in base] and [c.chunk_id for c in ctx] == [c.chunk_id for c in base]
    keep = next(c for c in ctx if "30일" in c.text)
    sec = next(c for c in ctx if "인증서" in c.text)
    assert keep.metadata["context"] == "[ops.md > 서버 운영 지침 > 3장 백업 > 3.2 보관 기간]"
    assert sec.metadata["context"] == "[ops.md > 서버 운영 지침 > 4장 보안 > 4.1 접속 절차]"
    assert all("context" not in c.metadata for c in base)        # 입력은 건드리지 않는다


def test_overlapping_length_chunks_are_located_and_documents_without_headings_get_filename_only():
    plain = "가나다라마바사 " * 80
    d = doc(plain, "scan.pdf")
    cs = add_context_headers(d, chunk_with_strategy(d, ChunkStrategyConfig("length", 100, 40)))
    assert len(cs) > 3 and {c.metadata["context"] for c in cs} == {"[scan.pdf]"}


def test_long_headers_are_capped_and_index_text_prepends_only_when_present():
    d = doc("# " + "아" * 90 + "\n\n## " + "b" * 90 + "\n\n본문 내용입니다 " * 5)
    c = add_context_headers(d, [Chunk("c", d.doc_id, "본문 내용입니다", "length", 0)])[0]
    assert len(c.metadata["context"]) <= 160 and c.metadata["context"].endswith("…]")
    assert index_text(Chunk("c", "d", "본문", "length", 0)) == "본문"
    assert index_text(c).endswith("\n본문 내용입니다") and index_text(c).startswith("[ops.md")


def test_bm25_finds_a_chunk_only_through_its_header_and_results_show_original_text():
    a = Chunk("a", "d1", "인증서를 설치해야 한다", "length", 0)
    b = Chunk("b", "d2", "백업은 매일 새벽에 수행한다", "length", 0)
    a_ctx = Chunk("a", "d1", a.text, "length", 0, {"context": "[vpn.md > 접속 절차]"})
    b_ctx = Chunk("b", "d2", b.text, "length", 0, {"context": "[backup.md > 보관 기간]"})
    assert BM25Index([a, b], tok_char_bigram, idf_variant="lucene").search("접속 절차", 2) == []   # 본문에 없는 말: 머리말 없이는 못 찾는다
    hits = BM25Index([a_ctx, b_ctx], tok_char_bigram, idf_variant="lucene").search("접속 절차", 2)
    assert [h.chunk_id for h in hits] == ["a"] and hits[0].text == "인증서를 설치해야 한다"      # 결과 본문에는 머리말이 섞이지 않는다


def test_grid_candidates_with_context_have_distinct_labels_and_search_works():
    d = doc()
    cfgs = [ChunkStrategyConfig("length", 80, 10), ChunkStrategyConfig("length", 80, 10, context_header=True)]
    assert [chunk_label(c) for c in cfgs] == ["length-80-10", "length-80-10+ctx"]
    cands = build_candidates([d], cfgs, [RetrieverSpec("bm25")])
    assert len(cands) == 2 and all(c.search("보관 기간", 2) for c in cands)
    hit = cands[1].search("보관 기간", 1)[0]
    assert "context" in hit.metadata and "[ops.md" not in hit.text

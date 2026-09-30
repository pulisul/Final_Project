from pathlib import Path
from intellectagent.chunk_strategies import ChunkStrategyConfig, chunk_with_strategy
from intellectagent.hashing import sha256_text
from intellectagent.models import Document


def _doc(text: str) -> Document:
    return Document("doc1", Path("x.md"), text, sha256_text(text), {"name": "x.md"})


def test_length_strategy_is_deterministic():
    doc = _doc("가" * 1800)
    cfg = ChunkStrategyConfig("length", size=700, overlap=80)
    a = chunk_with_strategy(doc, cfg)
    b = chunk_with_strategy(doc, cfg)
    assert [x.chunk_id for x in a] == [x.chunk_id for x in b]
    assert all(x.strategy == "length" for x in a)


def test_semantic_strategy_keeps_strategy_contract():
    doc = _doc("# 제목\n\n첫 문단입니다. " * 40 + "\n\n## 소제목\n\n두 번째 문단입니다. " * 40)
    out = chunk_with_strategy(doc, ChunkStrategyConfig("semantic"))
    assert out
    assert {x.strategy for x in out}.issubset({"semantic", "length"})


def test_unknown_strategy_rejected():
    doc = _doc("hello")
    try:
        chunk_with_strategy(doc, ChunkStrategyConfig("unknown"))
    except ValueError:
        return
    raise AssertionError("unknown strategy must be rejected")

from pathlib import Path
import pytest
from intellectagent.loaders import SafeDocumentLoader, UnsafePathError, UnsupportedFileError


def test_oversized_file_rejected(tmp_path: Path):
    root = tmp_path / "root"; root.mkdir()
    p = root / "big.txt"; p.write_bytes(b"x" * 101)
    loader = SafeDocumentLoader(root, (".txt",), 100)
    with pytest.raises(UnsupportedFileError):
        loader.load(p)


def test_symlink_rejected(tmp_path: Path):
    root = tmp_path / "root"; root.mkdir()
    target = root / "target.txt"; target.write_text("safe", encoding="utf-8")
    link = root / "link.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink is unavailable on this platform")
    loader = SafeDocumentLoader(root, (".txt",), 1024)
    with pytest.raises(UnsafePathError):
        loader.load(link)


def test_case_insensitive_extension_allowed(tmp_path: Path):
    root = tmp_path / "root"; root.mkdir()
    p = root / "A.TXT"; p.write_text("hello", encoding="utf-8")
    loader = SafeDocumentLoader(root, (".txt",), 1024)
    assert loader.load(p).text == "hello"


def test_expect_txt_memo_is_not_part_of_corpus(tmp_path: Path):
    """사람이 적은 '<문서>.expect.txt' 정답 메모가 말뭉치에 섞이면 질문 생성·평가가 오염된다."""
    from intellectagent.corpus_scan import scan_roots
    root = tmp_path / "root"; root.mkdir()
    (root / "a.txt").write_text("본문", encoding="utf-8")
    (root / "a.txt.expect.txt").write_text("정답 문장", encoding="utf-8")
    loader = SafeDocumentLoader(root, (".txt",), 10_000)
    assert [p.name for p in loader.discover()] == ["a.txt"]
    scan = scan_roots([root], (".txt",), 10_000, allow_home=True)
    assert [e.path.name for e in scan.entries] == ["a.txt"]

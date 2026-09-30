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

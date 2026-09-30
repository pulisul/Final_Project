from __future__ import annotations

from pathlib import Path

import pytest

from intellectagent.document_ingest import (
    UploadValidationError,
    list_ingestible_documents,
    normalize_upload_name,
    save_upload_bytes,
)


def test_upload_rejects_path_components():
    with pytest.raises(UploadValidationError):
        normalize_upload_name("../escape.txt")
    with pytest.raises(UploadValidationError):
        normalize_upload_name("folder/a.pdf")


def test_upload_rejects_extension_and_oversize(tmp_path: Path):
    with pytest.raises(UploadValidationError):
        save_upload_bytes(
            allowed_root=tmp_path,
            filename="bad.exe",
            data=b"abc",
            allowed_extensions=(".txt", ".md"),
            max_file_bytes=100,
        )
    with pytest.raises(UploadValidationError):
        save_upload_bytes(
            allowed_root=tmp_path,
            filename="large.txt",
            data=b"a" * 101,
            allowed_extensions=(".txt",),
            max_file_bytes=100,
        )


def test_upload_is_idempotent_and_never_silently_overwrites(tmp_path: Path):
    one = save_upload_bytes(
        allowed_root=tmp_path,
        filename="guide.txt",
        data="첫 번째 문서".encode(),
        allowed_extensions=(".txt",),
        max_file_bytes=1000,
    )
    duplicate = save_upload_bytes(
        allowed_root=tmp_path,
        filename="guide.txt",
        data="첫 번째 문서".encode(),
        allowed_extensions=(".txt",),
        max_file_bytes=1000,
    )
    changed = save_upload_bytes(
        allowed_root=tmp_path,
        filename="guide.txt",
        data="두 번째 문서".encode(),
        allowed_extensions=(".txt",),
        max_file_bytes=1000,
    )
    assert one.duplicate is False
    assert duplicate.duplicate is True
    assert changed.stored_name.startswith("guide__")
    assert (tmp_path / "guide.txt").read_text() == "첫 번째 문서"
    assert len(list_ingestible_documents(tmp_path, (".txt",))) == 2


def test_upload_rejects_empty_file(tmp_path: Path):
    with pytest.raises(UploadValidationError):
        save_upload_bytes(
            allowed_root=tmp_path,
            filename="empty.txt",
            data=b"",
            allowed_extensions=(".txt",),
            max_file_bytes=1000,
        )

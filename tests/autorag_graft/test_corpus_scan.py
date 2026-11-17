import os
import random
import sys
import time
from pathlib import Path

import pytest

from intellectagent import corpus_scan as cs
from intellectagent.config import CoreConfig
from intellectagent.doc_parsers import ParseError, extract_text
from intellectagent.document_ingest import make_loader
from intellectagent.loaders import UnsafePathError

EXTS = (".md", ".txt", ".pdf", ".eml")


def w(p: Path, text="내용 " * 20, mtime=None):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    if mtime:
        os.utime(p, (mtime, mtime))
    return p


def names(res):
    return sorted(e.path.name for e in res.entries)


def test_skips_system_secret_hidden_symlink_and_oversized(tmp_path):
    root = tmp_path / "docs"
    w(root / "ok.md"); w(root / "sub/ok2.txt")
    w(root / "node_modules/x.md"); w(root / ".git/y.md"); w(root / ".ssh/z.txt"); w(root / "build/b.md"); w(root / ".hidden/h.md")
    w(root / ".env.txt"); w(root / "비밀번호 모음.txt"); w(root / "credentials.md"); w(root / "id_rsa.txt"); w(root / ".dot.md"); w(root / "~$lock.md")
    (root / "empty.md").write_text("")
    big = w(root / "big.md", "x" * 5000)
    outside = tmp_path / "outside"; w(outside / "leak.md")
    (root / "link_dir").symlink_to(outside, target_is_directory=True)
    (root / "link.md").symlink_to(outside / "leak.md")
    res = cs.scan_roots([root], EXTS, max_file_bytes=1000)
    assert names(res) == ["ok.md", "ok2.txt"]
    assert res.skipped["크기 제한 초과"] == 1 and res.skipped["빈 파일"] == 1 and res.skipped["심볼릭 링크 폴더"] == 1 and res.skipped["심볼릭 링크 파일"] == 1
    assert res.skipped["비밀번호·키로 보이는 파일명"] == 3 and res.skipped["숨김/임시 파일"] == 3 and res.skipped["제외 폴더(숨김/시스템/캐시/자격증명)"] >= 5
    assert not any("leak" in str(e.path) for e in res.entries)


def test_dangerous_roots_are_rejected(tmp_path, monkeypatch):
    for bad in (Path(Path.cwd().anchor), Path("/proc") if sys.platform == "linux" else tmp_path / "nope", tmp_path / "missing"):
        with pytest.raises(cs.RootRejected):
            cs.validate_root(bad)
    home = tmp_path / "home"; (home / "Documents").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    with pytest.raises(cs.RootRejected):
        cs.validate_root(home)
    assert cs.validate_root(home, allow_home=True) == home.resolve()
    assert cs.validate_root(home / "Documents") == (home / "Documents").resolve()
    (tmp_path / "node_modules/pkg").mkdir(parents=True)
    with pytest.raises(cs.RootRejected):
        cs.validate_root(tmp_path / "node_modules/pkg")


def test_overlapping_roots_do_not_duplicate_files(tmp_path):
    w(tmp_path / "a/b/f.md")
    res = cs.scan_roots([tmp_path / "a", tmp_path / "a/b"], EXTS, 10**6)
    assert names(res) == ["f.md"] and len(res.roots) == 1


def test_index_and_cache_dirs_are_excluded(tmp_path):
    root = tmp_path / "docs"; w(root / "a.md"); w(root / "idx/mirror.md"); w(root / "ev/e.md")
    res = cs.scan_roots([root], EXTS, 10**6, exclude_paths=[root / "idx", root / "ev"])
    assert names(res) == ["a.md"]


def test_dedupe_keeps_newest_copy_deterministically_and_hashes_only_same_size(tmp_path, monkeypatch):
    now = time.time()
    w(tmp_path / "old/a.md", "같은 내용입니다 " * 10, mtime=now - 1000)
    w(tmp_path / "new/a_copy.md", "같은 내용입니다 " * 10, mtime=now)
    w(tmp_path / "x/different.md", "다른 내용 " * 11)
    w(tmp_path / "y/same_size_other.md", "같은 내용입니다 " * 9 + "다른 마지막 줄!!", mtime=now)
    calls = []
    real = cs._sha256_file
    monkeypatch.setattr(cs, "_sha256_file", lambda p: calls.append(p.name) or real(p))
    res = cs.scan_roots([tmp_path], EXTS, 10**6)
    kept, dropped = cs.dedupe_exact(res.entries)
    assert "a_copy.md" in names(type("R", (), {"entries": kept})) and "a.md" not in [e.path.name for e in kept]
    assert [(d.path.name, k.path.name) for d, k in dropped] == [("a.md", "a_copy.md")]
    assert "different.md" not in calls          # 크기가 유일한 파일은 읽지 않는다
    k2, _ = cs.dedupe_exact(list(reversed(res.entries)))
    assert [e.path for e in k2] == [e.path for e in kept]


def test_manifest_diff_detects_added_modified_removed(tmp_path):
    f1, f2 = w(tmp_path / "a.md", "aaaa " * 10), w(tmp_path / "b.md", "bbbb " * 10)
    old = cs.build_manifest(cs.scan_roots([tmp_path], EXTS, 10**6).entries)
    cs.save_manifest(tmp_path.parent / "m" / "man.json", old)
    f2.write_text("bbbb changed " * 10, encoding="utf-8"); f1.unlink(); w(tmp_path / "c.md", "cccc " * 10)
    d = cs.diff_manifest(cs.load_manifest(tmp_path.parent / "m" / "man.json"), cs.build_manifest(cs.scan_roots([tmp_path], EXTS, 10**6).entries))
    assert [Path(x).name for x in d["added"]] == ["c.md"] and [Path(x).name for x in d["removed"]] == ["a.md"] and [Path(x).name for x in d["modified"]] == ["b.md"]
    assert cs.load_manifest(tmp_path / "nope.json") == {}


def test_suggest_counts_without_following_symlinks(tmp_path):
    home = tmp_path / "h"; w(home / "Documents/a.md"); w(home / "Documents/.git/x.md"); (home / "Desktop").mkdir()
    out = cs.suggest_candidate_roots(EXTS, home=home)
    assert {Path(c["path"]).name: c["supported_files"] for c in out} == {"Documents": 1, "Desktop": 0}


def test_multi_root_loader_loads_inside_rejects_outside_and_symlink(tmp_path):
    a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
    w(a / "1.md", "알파 문서 " * 10); w(b / "2.md", "베타 문서 " * 10); w(out / "3.md"); 
    ld = cs.MultiRootLoader([a, b], EXTS, 10**6)
    assert sorted(p.name for p in ld.discover()) == ["1.md", "2.md"]
    assert ld.load(b / "2.md").text.startswith("베타") and ld.load(a / "1.md").metadata["name"] == "1.md"
    with pytest.raises(UnsafePathError):
        ld.load(out / "3.md")
    (a / "ln.md").symlink_to(out / "3.md")
    with pytest.raises(UnsafePathError):
        ld.load(a / "ln.md")
    with pytest.raises(UnsafePathError):
        ld.load(a / ".." / "out" / "3.md")


def test_make_loader_uses_search_roots_and_keeps_cache_outside(tmp_path):
    main, extra = tmp_path / "docs", tmp_path / "more"
    w(main / "a.md", "본문 하나 " * 10); w(extra / "b.md", "본문 둘 " * 10); w(extra / "b_dup.md", "본문 둘 " * 10, mtime=time.time() - 500)
    cfg = CoreConfig(project_root=tmp_path, allowed_root=main, index_dir=tmp_path / "idx", metadata_db=tmp_path / "m.db", evidence_dir=tmp_path / "ev",
                     allowed_extensions=(".md",), feature_flags={"search_roots": [str(extra)], "use_parse_cache": True})
    ld = make_loader(cfg)
    assert sorted(p.name for p in ld.discover()) == ["a.md", "b.md"]       # b_dup 은 더 오래된 중복이라 제외
    assert ld.load(extra / "b.md").text.startswith("본문 둘")
    cfg2 = CoreConfig(project_root=tmp_path, allowed_root=main, index_dir=main / "idx", metadata_db=tmp_path / "m.db", evidence_dir=tmp_path / "ev",
                      allowed_extensions=(".md",), feature_flags={"search_roots": [str(extra)], "use_parse_cache": True})
    with pytest.raises(ValueError):
        make_loader(cfg2)       # 캐시가 문서 루트 안이면 거부
    cfg3 = CoreConfig(project_root=tmp_path, allowed_root=main, index_dir=tmp_path / "idx", metadata_db=tmp_path / "m.db", evidence_dir=tmp_path / "ev",
                      allowed_extensions=(".md",), feature_flags={})
    assert type(make_loader(cfg3)).__name__ == "SafeDocumentLoader"          # 기본값은 기존 동작 그대로


EML = ("From: =?utf-8?b?7ZmN6ri47IiY?= <hong@example.com>\nTo: me@example.com\nSubject: =?utf-8?b?7ISc67KEIOuwseyXhSDsnbzsoJU=?=\n"
       "Date: Mon, 19 Oct 2026 09:00:00 +0900\nMIME-Version: 1.0\nContent-Type: multipart/mixed; boundary=B\n\n--B\n"
       "Content-Type: text/html; charset=utf-8\n\n<html><style>p{}</style><body><p>백업은 <b>새벽 2시</b></p><script>x()</script></body></html>\n--B\n"
       "Content-Type: application/pdf; name=\"계획서.pdf\"\nContent-Disposition: attachment; filename=\"계획서.pdf\"\n\nJVBERi0=\n--B--\n").encode()


def test_eml_headers_html_body_and_attachment_name():
    t = extract_text(Path("m.eml"), EML)
    assert "제목: 서버 백업 일정" in t and "보낸 사람: 홍길수" in t and "새벽 2시" in t and "[첨부] 계획서.pdf" in t
    assert "x()" not in t and "p{}" not in t and "<b>" not in t


def test_eml_plain_preferred_and_garbage_never_leaks_other_exceptions():
    plain = b"Subject: hi\nContent-Type: text/plain; charset=euc-kr\n\n" + "안녕하세요 본문".encode("euc-kr")
    assert "안녕하세요 본문" in extract_text(Path("p.eml"), plain)
    rng = random.Random(7)
    for _ in range(300):
        raw = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 300)))
        try:
            extract_text(Path("g.eml"), raw)
        except ParseError:
            pass
    for i in range(0, len(EML), 7):
        try:
            extract_text(Path("t.eml"), EML[:i])
        except ParseError:
            pass


def test_cli_refuses_without_roots_and_reports(tmp_path, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("sc", Path(__file__).parents[2] / "scripts/autorag/scan_corpus.py")
    m = importlib.util.module_from_spec(spec); sys.modules["sc"] = m; spec.loader.exec_module(m)
    assert m.main([]) == 2
    root = tmp_path / "r"; w(root / "a.md"); w(root / "b.md"); w(root / "password.txt")
    mf = tmp_path / "man.json"
    assert m.main(["--roots", str(root), "--manifest", str(mf)]) == 0
    out = capsys.readouterr().out
    assert "색인 대상 1개" in out and "중복 제외 1개" in out and "비밀번호" in out and "첫 실행" in out
    w(root / "c.md", "새 파일 " * 10)
    assert m.main(["--roots", str(root), "--manifest", str(mf), "--json"]) == 0
    import json
    rep = json.loads(capsys.readouterr().out)
    assert rep["changes_since_last"] == {"added": 1, "removed": 0, "modified": 0}
    assert m.main(["--roots", "/"]) == 2


def test_two_folders_one_index_end_to_end_with_mail_and_incremental_add(tmp_path):
    from intellectagent.factory import build_components
    from intellectagent.indexing import Indexer
    from intellectagent.metadata import MetadataDB

    main, mail = tmp_path / "docs", tmp_path / "mail"
    w(main / "backup.md", "서버 백업 지침: 매일 새벽 2시에 전체 백업을 수행하고 30일간 보관한다. 복구 훈련은 분기마다 실시한다. " * 3)
    mail.mkdir()
    (mail / "herb.eml").write_bytes("Subject: 황기 재배 일정\nContent-Type: text/plain; charset=utf-8\n\n황기와 당귀는 배수가 잘되는 토양에서 봄에 파종한다고 공유드립니다. 파종 간격은 삼십 센티미터입니다.\n".encode())
    cfg = CoreConfig(project_root=tmp_path, allowed_root=main, index_dir=tmp_path / "idx", metadata_db=tmp_path / "m.db", evidence_dir=tmp_path / "ev",
                     allowed_extensions=(".md", ".eml"), semantic_min_chars=20, semantic_max_chars=200,
                     feature_flags={"search_roots": [str(mail)], "retrieval_mode": "hybrid_rrf"})
    emb, store, retr, gen, pipe = build_components(cfg)
    loader = make_loader(cfg)
    Indexer(loader, emb, store, MetadataDB(cfg.metadata_db), cfg).index_all(tmp_path / "ck.json")
    assert not pipe.ask("황기는 언제 파종하나요").fallback and "황기" in pipe.ask("황기는 언제 파종하나요").hits[0].text
    assert "백업" in pipe.ask("서버 백업은 언제 하나요").hits[0].text
    (mail / "vpn.eml").write_bytes("Subject: VPN\nContent-Type: text/plain; charset=utf-8\n\n원격 근무 VPN 접속은 사내 인증서를 설치한 뒤 이중 인증을 통과해야 합니다. 문의는 보안팀으로 해주세요.\n".encode())
    Indexer(loader, emb, store, MetadataDB(cfg.metadata_db), cfg).index_all(tmp_path / "ck.json")
    assert "VPN" in pipe.ask("VPN 접속 방법").hits[0].text      # 새로 생긴 메일이 BM25/벡터 양쪽에 자동 반영

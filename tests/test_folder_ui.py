"""폴더 찾기 화면 로직(folder_ui) 테스트. 모델 없이 임시 폴더만 쓴다."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from intellectagent import folder_ui as fu
from intellectagent.config import CoreConfig
from intellectagent.corpus_scan import RootRejected

EXTS = (".md", ".txt")


def make_cfg(tmp: Path) -> CoreConfig:
    (tmp / "docs0").mkdir(exist_ok=True)
    return CoreConfig(project_root=tmp, allowed_root=tmp / "docs0", index_dir=tmp / "index" / "chroma",
                      metadata_db=tmp / "index" / "m.sqlite3", evidence_dir=tmp / "evidence",
                      allowed_extensions=EXTS, max_file_bytes=10_000, feature_flags={"x": 1})


def folder(tmp: Path, name="mydocs") -> Path:
    d = tmp / name
    (d / "sub").mkdir(parents=True)
    (d / "a.md").write_text("백업은 매일 02:00 에 합니다.", encoding="utf-8")
    (d / "b.txt").write_text("휴가 규정 문서", encoding="utf-8")
    (d / "sub" / "c.md").write_text("서브폴더 문서", encoding="utf-8")
    (d / "dup.md").write_text("백업은 매일 02:00 에 합니다.", encoding="utf-8")      # a.md 와 내용이 같음
    (d / "image.png").write_bytes(b"x")                                                    # 지원하지 않는 확장자
    (d / "empty.md").write_text("", encoding="utf-8")
    (d / "big.md").write_text("가" * 20_000, encoding="utf-8")                            # 크기 제한(10,000B) 초과
    (d / "password.txt").write_text("비밀", encoding="utf-8")                              # 비밀번호로 보이는 이름
    (d / ".hidden").mkdir(); (d / ".hidden" / "h.md").write_text("숨김", encoding="utf-8")
    return d


def test_scan_counts_dedupes_and_explains_skips(tmp_path):
    cfg = make_cfg(tmp_path); d = folder(tmp_path)
    s = fu.scan_folder(cfg, d)
    assert s.files == 4 and s.unique_files == 3 and s.duplicates == 1       # a, b, c, dup → dup 은 a 와 같은 내용
    assert s.by_ext == {".md": 2, ".txt": 1}
    for reason in ("지원하지 않는 확장자", "빈 파일", "크기 제한 초과", "비밀번호·키로 보이는 파일명"):
        assert s.skipped.get(reason) == 1, (reason, s.skipped)
    assert any("숨김" in k or "제외 폴더" in k for k in s.skipped)
    assert sorted(s.samples) == sorted(set(s.samples)) and all(not Path(x).is_absolute() for x in s.samples)
    assert s.total_mb >= 0 and not s.truncated


def test_scan_does_not_index_or_write(tmp_path):
    cfg = make_cfg(tmp_path); d = folder(tmp_path)
    before = sorted(p.name for p in tmp_path.rglob("*"))
    fu.scan_folder(cfg, d)
    assert sorted(p.name for p in tmp_path.rglob("*")) == before


@pytest.mark.parametrize("text", ["", "   ", "/", "/etc", "/usr/bin"])
def test_dangerous_or_empty_roots_are_rejected(tmp_path, text):
    with pytest.raises(RootRejected):
        fu.check_root(make_cfg(tmp_path), text)


def test_home_and_missing_and_file_are_rejected(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path); f = tmp_path / "f.md"; f.write_text("x", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "home")); (tmp_path / "home").mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    for bad in (str(tmp_path / "home"), str(tmp_path / "nope"), str(f)):
        with pytest.raises(RootRejected):
            fu.check_root(cfg, bad)


def test_own_index_and_evidence_folders_are_rejected_but_parent_is_allowed(tmp_path):
    cfg = make_cfg(tmp_path); (tmp_path / "index" / "chroma" / "deep").mkdir(parents=True); (tmp_path / "evidence").mkdir()
    for bad in (cfg.index_dir, cfg.index_dir / "deep", cfg.evidence_dir):
        with pytest.raises(RootRejected):
            fu.check_root(cfg, str(bad))
    assert fu.check_root(cfg, str(tmp_path)) == tmp_path.resolve()        # 상위는 허용(색인에서 자동 제외됨)


def test_scan_of_parent_excludes_own_index_and_evidence(tmp_path):
    cfg = make_cfg(tmp_path); (tmp_path / "index" / "chroma").mkdir(parents=True); (tmp_path / "evidence").mkdir()
    (tmp_path / "index" / "chroma" / "leak.md").write_text("내부", encoding="utf-8")
    (tmp_path / "evidence" / "ev.md").write_text("증거", encoding="utf-8")
    (tmp_path / "real.md").write_text("진짜 문서", encoding="utf-8")
    s = fu.scan_folder(cfg, tmp_path)
    assert s.samples == ["real.md"] and s.unique_files == 1


def test_add_root_persists_dedupes_and_revalidates(tmp_path):
    cfg = make_cfg(tmp_path); d = folder(tmp_path)
    r, new = fu.add_root(cfg, str(d)); assert new and r == d.resolve()
    assert fu.add_root(cfg, str(d))[1] is False                              # 두 번째는 추가 아님
    roots, notes = fu.load_roots(cfg); assert roots == [d.resolve()] and notes == []
    assert json.loads(fu.roots_file(cfg).read_text(encoding="utf-8")) == {"roots": [str(d.resolve())]}
    import shutil; shutil.rmtree(d)                                          # 폴더가 사라지면 건너뛰고 알린다
    roots, notes = fu.load_roots(cfg); assert roots == [] and len(notes) == 1


def test_load_roots_ignores_corrupt_and_dangerous_entries(tmp_path):
    cfg = make_cfg(tmp_path); p = fu.roots_file(cfg); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{깨짐", encoding="utf-8"); roots, notes = fu.load_roots(cfg); assert roots == [] and notes
    good = folder(tmp_path)
    p.write_text(json.dumps({"roots": ["/etc", str(good)]}), encoding="utf-8")   # 파일을 손으로 고쳐 위험 경로를 넣어도 거부
    roots, notes = fu.load_roots(cfg); assert roots == [good.resolve()] and len(notes) == 1


def test_apply_roots_returns_copy_and_keeps_other_flags(tmp_path):
    cfg = make_cfg(tmp_path); d = folder(tmp_path)
    out = fu.apply_roots(cfg, [d.resolve()])
    assert out.feature_flags == {"x": 1, "search_roots": [str(d.resolve())]} and cfg.feature_flags == {"x": 1}
    assert fu.apply_roots(cfg, []).feature_flags == {"x": 1}


def test_registered_documents_uses_base_and_registered_roots(tmp_path):
    cfg = make_cfg(tmp_path); d = folder(tmp_path); (cfg.allowed_root / "base.md").write_text("기본 문서", encoding="utf-8")
    names = sorted(p.name for p in fu.registered_documents(fu.apply_roots(cfg, [d.resolve()])))
    assert len(names) == 4 and {"b.txt", "base.md", "c.md"} <= set(names)
    assert len({"a.md", "dup.md"} & set(names)) == 1                        # 내용이 같은 두 파일 중 하나만
    assert [p.name for p in fu.registered_documents(cfg)] == ["base.md"]


def test_suggest_folders_lists_only_folders_with_documents(tmp_path, monkeypatch):
    home = tmp_path / "home"; (home / "Documents").mkdir(parents=True); (home / "Downloads").mkdir()
    (home / "Documents" / "x.md").write_text("문서", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    out = fu.suggest_folders(make_cfg(tmp_path))
    assert [Path(o["path"]).name for o in out] == ["Documents"] and out[0]["supported_files"] == 1

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from intellectagent.doc_check import (check_expected, collect_files, hwp_diagnostics, hwpx_diagnostics, inspect_file, normalize_for_match,
                                      read_expected, snippets, summarize, text_quality)
from .hwp_fixture import make_hwp
from .test_doc_parsers import make_hwpx

SPEC = importlib.util.spec_from_file_location("crd", Path(__file__).parents[2] / "scripts/autorag/check_real_docs.py")
crd = importlib.util.module_from_spec(SPEC); sys.modules["crd"] = crd; SPEC.loader.exec_module(crd)

GOOD = "연차휴가는 입사 후 1년이 지나면 15일이 부여된다. 병가는 연 60일까지 사용할 수 있다.\n" * 3


def test_text_quality_clean_korean_has_no_flags():
    q = text_quality(GOOD)
    assert q["flags"] == [] and q["hangul_ratio"] > 0.5


def test_text_quality_flags():
    assert text_quality("   \n")["flags"] == ["EMPTY"]
    assert "REPLACEMENT_CHARS" in text_quality("가나다라" * 10 + "�" * 5)["flags"]
    assert "CJK_WITHOUT_HANGUL" in text_quality("漢字汉字文書規定" * 20)["flags"]            # 모지바케 전형
    assert "ONE_WORD_PER_LINE" in text_quality("\n".join(f"단어{i}" for i in range(40)))["flags"]
    assert "MANY_DUPLICATE_LINES" in text_quality("머리글 반복 문장입니다\n" * 30 + "본문 하나\n")["flags"]
    assert "PRIVATE_USE_HIGH" in text_quality("가" * 20 + "" * 5)["flags"]
    assert "CONTROL_CHARS" in text_quality("가나다" * 20 + "\x01" * 3)["flags"]


def test_check_expected_ignores_whitespace_and_newline_differences():
    text = "연차휴가는 입사 후\n1년이 지나면   15일이\n부여된다."
    r = check_expected(text, ["연차휴가는 입사 후 1년이 지나면 15일이 부여된다", "존재하지 않는 문장"])
    assert r["found"] == 1 and r["missing"] == ["존재하지 않는 문장"] and r["recall"] == 0.5
    assert check_expected("x", [])["recall"] is None
    assert normalize_for_match("가 나\t다") == "가나다"


def test_nfd_text_matches_nfc_phrase():
    import unicodedata
    nfd = unicodedata.normalize("NFD", "한글 문서")
    assert check_expected(nfd, ["한글 문서"])["found"] == 1


def test_read_expected_sidecar(tmp_path):
    f = tmp_path / "a.txt"; f.write_text("x", encoding="utf-8")
    (tmp_path / "a.txt.expect.txt").write_text("# 주석\n\n문장 하나\n 문장 둘 \n", encoding="utf-8")
    assert read_expected(f) == ["문장 하나", "문장 둘"]
    d = tmp_path / "exp"; d.mkdir(); (d / "a.txt.expect.txt").write_text("다른 문장", encoding="utf-8")
    assert read_expected(f, d) == ["다른 문장"]                   # expect-dir 가 우선
    assert read_expected(tmp_path / "none.txt") == []


def test_hwp_diagnostics_flags():
    pytest.importorskip("olefile")
    d = hwp_diagnostics(make_hwp([["안녕"]], compressed=True))
    assert d["signature_ok"] and d["compressed"] and not d["encrypted"] and d["sections"] == 1
    d2 = hwp_diagnostics(make_hwp([["안녕"]], compressed=False, encrypted=True, distribution=True))
    assert d2["encrypted"] and d2["distribution"] and not d2["compressed"]
    assert "error" in hwp_diagnostics(b"not an ole file")


def test_hwpx_diagnostics():
    d = hwpx_diagnostics(make_hwpx({"Contents/section0.xml": ["가"], "Contents/section1.xml": ["나"]}))
    assert d["sections"] == 2 and d["format"] == "hwpx"
    assert "error" in hwpx_diagnostics(b"nope")


def test_snippets_cover_head_middle_tail():
    s = snippets("가" * 50 + "나" * 50 + "다" * 50, 3, 10)
    assert len(s) == 3 and s[0].startswith("가") and "나" in s[1] and s[2].endswith("다")
    assert snippets("", 3) == [] and snippets("abc", 3) == ["abc"] and snippets("abc" * 20, 0) == []


def test_inspect_file_status_matrix(tmp_path):
    good = tmp_path / "g.md"; good.write_text(GOOD, encoding="utf-8")
    r = inspect_file(good)
    assert r["status"] == "OK_UNVERIFIED" and r["expected"] is None
    assert inspect_file(good, expect=["병가는 연 60일까지 사용할 수 있다"])["status"] == "OK"
    s = inspect_file(good, expect=["없는 문장"])
    assert s["status"] == "SUSPECT" and any(p.startswith("EXPECTED_MISSING") for p in s["problems"])
    empty = tmp_path / "e.txt"; empty.write_text("  \n", encoding="utf-8")
    assert inspect_file(empty)["status"] == "FAIL"
    broken = tmp_path / "x.hwpx"; broken.write_bytes(b"garbage")
    rb = inspect_file(broken)
    assert rb["status"] == "FAIL" and rb["error"]                  # 예외를 던지지 않고 사유를 기록


def test_inspect_real_hwp_and_hwpx_fixtures_with_expected(tmp_path):
    pytest.importorskip("olefile")
    h = tmp_path / "a.hwp"; h.write_bytes(make_hwp([["연차휴가는 15일이다.", "병가는 60일이다."]]))
    r = inspect_file(h, expect=["연차휴가는 15일이다", "병가는 60일이다"])
    assert r["status"] == "OK" and r["diagnostics"]["compressed"]
    x = tmp_path / "a.hwpx"; x.write_bytes(make_hwpx({"Contents/section0.xml": ["연차휴가는 15일이다."]}))
    assert inspect_file(x, expect=["연차휴가는 15일이다"])["status"] == "OK"
    assert inspect_file(x, expect=["엉뚱한 문장"])["status"] == "SUSPECT"      # 기대 문장이 없으면 잡아낸다


def test_dump_dir_writes_text_but_report_has_no_body_by_default(tmp_path):
    f = tmp_path / "g.md"; f.write_text(GOOD, encoding="utf-8")
    r = inspect_file(f, dump_dir=tmp_path / "dump")
    assert (tmp_path / "dump/g.md.extracted.txt").read_text(encoding="utf-8").startswith("연차휴가")
    assert "head" not in r and "samples" not in r and "연차휴가" not in json.dumps(r, ensure_ascii=False)


def test_collect_files_skips_expect_sidecars_and_unsupported(tmp_path):
    (tmp_path / "a.md").write_text("x", encoding="utf-8"); (tmp_path / "a.md.expect.txt").write_text("x", encoding="utf-8")
    (tmp_path / "b.exe").write_bytes(b"x"); (tmp_path / "sub").mkdir(); (tmp_path / "sub/c.txt").write_text("x", encoding="utf-8")
    assert [p.name for p in collect_files([tmp_path])] == ["a.md", "c.txt"]


def test_summarize_counts():
    s = summarize([{"ext": ".hwp", "status": "OK", "expected": {"total": 2}}, {"ext": ".hwp", "status": "FAIL"}, {"ext": ".pdf", "status": "OK_UNVERIFIED"}])
    assert s["by_status"] == {"OK": 1, "FAIL": 1, "OK_UNVERIFIED": 1} and s["verified_files"] == 1 and s["verified_ok"] == 1


def test_script_exit_codes_and_privacy(tmp_path, capsys):
    d = tmp_path / "docs"; d.mkdir()
    (d / "g.md").write_text(GOOD, encoding="utf-8")
    out = tmp_path / "out"
    assert crd.main([str(d), "--out", str(out)]) == 0                                  # OK_UNVERIFIED 는 통과(증거 아님은 보고서에 표시)
    rep = (out / "report.md").read_text(encoding="utf-8") + (out / "report.json").read_text(encoding="utf-8")
    assert "연차휴가" not in rep and "OK_UNVERIFIED" in rep
    (d / "g.md.expect.txt").write_text("없는 문장\n", encoding="utf-8")
    assert crd.main([str(d)]) == 1                                                      # 기대 문장 누락 → 실패 코드
    assert crd.main([str(tmp_path / "nothing")]) == 2

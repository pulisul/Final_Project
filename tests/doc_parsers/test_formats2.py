"""실문서 점검에서 나온 추가분: PDF 정리, OpenDocument(.odt/.ods/.odp), 구형 Word(.doc), 구형 PowerPoint(.ppt)."""
import io
import random
import struct
import time
import zipfile
from pathlib import Path

import pytest

from intellectagent import doc_parsers as dp
from intellectagent.doc_parsers import ParseError, clean_pdf_page, extract_text
from intellectagent.loaders import SafeDocumentLoader

from . import binary_fixtures as F

NS = ('xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
      'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" '
      'xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0"')


def odf(kind: str, body: str, manifest: str = "<manifest/>", doctype: str = "") -> bytes:
    xml = f'<?xml version="1.0" encoding="UTF-8"?>{doctype}<office:document-content {NS}><office:body><office:{kind}>{body}</office:{kind}></office:body></office:document-content>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("mimetype", f"application/vnd.oasis.opendocument.{kind}")
        z.writestr("META-INF/manifest.xml", manifest)
        z.writestr("content.xml", xml)
    return buf.getvalue()


# ---- PDF 텍스트 정리 --------------------------------------------------------------------------------------------------
def test_clean_pdf_reflows_word_per_line_extraction_and_fixes_ligatures():
    words = "The permanent and ofﬁcial location for the working group is online".split()
    raw = " \n".join(["\n".join([w, " "]) for w in words]) + "\n \n \nSecond paragraph starts here with ﬁndings"
    out = clean_pdf_page(raw)
    assert "official" in out and "ﬁ" not in out and "findings" in out
    assert out.split("\n\n")[0].startswith("The permanent and official location")   # 낱말이 한 줄로 이어짐
    assert "\n\n" in out                                                              # 큰 간격은 문단 경계로 남음


def test_clean_pdf_leaves_normal_pages_alone_except_ligatures_and_trailing_space():
    page = "서버 백업은 매일 새벽 두 시에 실행되며 사본은 다른 지역에 보관합니다.   \n휴가 신청은 팀장 승인을 거쳐 인사시스템에서 합니다.\n\n새 문단입니다 oﬃce"
    assert clean_pdf_page(page) == "서버 백업은 매일 새벽 두 시에 실행되며 사본은 다른 지역에 보관합니다.\n휴가 신청은 팀장 승인을 거쳐 인사시스템에서 합니다.\n\n새 문단입니다 office"


def test_clean_pdf_keeps_short_line_slides_that_are_not_blank_separated():
    page = "\n".join(["목표", "일정", "예산", "위험", "범위", "담당", "산출물", "검수", "종료"])   # 짧지만 빈 줄 없이 줄바꿈만
    assert clean_pdf_page(page) == page


def test_clean_pdf_strips_nul_and_handles_empty():
    assert clean_pdf_page("a\x00b") == "ab" and clean_pdf_page("") == ""


# ---- OpenDocument ----------------------------------------------------------------------------------------------------
def test_odt_text_spaces_tabs_breaks_lists_tables_footnotes_and_skips_comments():
    body = ("<text:h>서버 지침</text:h>"
            '<text:p>서버<text:s text:c="3"/>백업<text:tab/>매일<text:line-break/>다음줄<text:note><text:note-citation>1</text:note-citation>'
            '<text:note-body><text:p>각주 내용</text:p></text:note-body></text:note></text:p>'
            "<text:list><text:list-item><text:p>항목1</text:p><text:list><text:list-item><text:p>하위항목</text:p></text:list-item></text:list></text:list-item></text:list>"
            "<table:table><table:table-row><table:table-cell><text:p>항목</text:p></table:table-cell><table:table-cell><text:p>기한</text:p></table:table-cell></table:table-row>"
            "<table:table-row><table:table-cell><text:p>영수증</text:p></table:table-cell><table:table-cell><text:p>7일</text:p></table:table-cell></table:table-row></table:table>"
            "<text:p><office:annotation><text:p>비공개 메모</text:p></office:annotation>본문<text:bookmark text:name='x'/></text:p>")
    out = extract_text(Path("a.odt"), odf("text", body))
    assert out.splitlines() == ["서버 지침", "서버   백업\t매일", "다음줄[각주: 각주 내용]", "항목1", "하위항목", "항목 | 기한", "영수증 | 7일", "본문"]
    assert "비공개 메모" not in out


def test_ods_sheets_and_huge_empty_repeats_do_not_explode():
    body = ('<table:table table:name="예산"><table:table-row><table:table-cell><text:p>항목</text:p></table:table-cell><table:table-cell><text:p>금액</text:p></table:table-cell></table:table-row>'
            '<table:table-row><table:table-cell><text:p>서버</text:p></table:table-cell><table:table-cell table:number-columns-repeated="1000000"/></table:table-row>'
            '<table:table-row table:number-rows-repeated="1048576"><table:table-cell table:number-columns-repeated="16384"/></table:table-row></table:table>'
            '<table:table table:name="메모"><table:table-row><table:table-cell table:number-columns-repeated="3"><text:p>반복</text:p></table:table-cell></table:table-row></table:table>')
    t = time.perf_counter()
    out = extract_text(Path("a.ods"), odf("spreadsheet", body))
    assert time.perf_counter() - t < 1.0
    assert out.splitlines() == ["[시트 예산]", "항목 | 금액", "서버", "[시트 메모]", "반복 | 반복 | 반복"]


def test_ods_covered_cells_keep_columns_aligned():
    body = ('<table:table table:name="s"><table:table-row><table:table-cell><text:p>a</text:p></table:table-cell><table:covered-table-cell/>'
            "<table:table-cell><text:p>c</text:p></table:table-cell></table:table-row></table:table>")
    assert extract_text(Path("a.ods"), odf("spreadsheet", body)).splitlines() == ["[시트 s]", "a |  | c"]


def test_odp_slides_notes_and_skips_slide_number_frames():
    page = lambda n, t, note: (f'<draw:page draw:name="p{n}"><draw:frame><draw:text-box><text:p>{t}</text:p></draw:text-box></draw:frame>'
                               '<draw:frame presentation:class="page-number"><draw:text-box><text:p><text:page-number>&lt;number&gt;</text:page-number></text:p></draw:text-box></draw:frame>'
                               f'<presentation:notes><draw:frame><draw:text-box><text:p>{note}</text:p></draw:text-box></draw:frame>'
                               '<draw:frame presentation:class="page-number"><draw:text-box><text:p>&lt;number&gt;</text:p></draw:text-box></draw:frame></presentation:notes></draw:page>')
    out = extract_text(Path("a.odp"), odf("presentation", page(1, "개요", "첫 노트") + page(2, "마무리", "")))
    assert out.splitlines() == ["[슬라이드 1]", "개요", "[발표자 노트]", "첫 노트", "[슬라이드 2]", "마무리"]
    assert "<number>" not in out


def test_odf_rejects_entity_bombs_encrypted_and_garbage():
    bomb = '<!DOCTYPE x [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;&a;">]>'
    with pytest.raises(ParseError, match="DOCTYPE|ENTITY"):
        extract_text(Path("a.odt"), odf("text", "<text:p>&b;</text:p>", doctype=bomb))
    with pytest.raises(ParseError, match="암호"):
        extract_text(Path("a.odt"), odf("text", "<text:p>x</text:p>", manifest="<manifest><encryption-data/></manifest>"))
    with pytest.raises(ParseError):
        extract_text(Path("a.odt"), b"not a zip")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("mimetype", "x")
    with pytest.raises(ParseError, match="content.xml"):
        extract_text(Path("a.ods"), buf.getvalue())


# ---- .doc ------------------------------------------------------------------------------------------------------------
def doc_bytes() -> bytes:
    return F.load(F.FX_DOC)


def test_doc_extracts_paragraphs_table_and_normalizes_ligature():
    out = extract_text(Path("fx.doc"), doc_bytes())
    assert out.splitlines() == ["서버 운영 지침", "서버 백업은 매일 새벽 2시에 자동 실행되며 사본은 30일간 보관합니다.",
                                "Official release: the first deployment is verified by the on-call engineer.", "항목 | 기한", "영수증 제출 | 7일 이내"]


def test_doc_masquerading_formats_and_unsupported_versions():
    docx = pytest.importorskip("docx")
    d = docx.Document(); d.add_paragraph("실제로는 docx 입니다"); b = io.BytesIO(); d.save(b)
    assert "실제로는 docx 입니다" in extract_text(Path("x.doc"), b.getvalue())       # 확장자만 .doc 인 docx
    with pytest.raises(ParseError, match="RTF"):
        extract_text(Path("x.doc"), b"{\\rtf1\\ansi hello}")
    with pytest.raises(ParseError):
        extract_text(Path("x.doc"), b"plain text pretending to be doc")


def _patched_ole(tmp_path: Path, raw: bytes, stream: str, patch) -> bytes:
    olefile = pytest.importorskip("olefile")
    f = tmp_path / "x.ole"; f.write_bytes(raw)
    ole = olefile.OleFileIO(str(f), write_mode=True)
    try:
        data = bytearray(ole.openstream(stream).read()); patch(data)
        ole.write_stream(stream, bytes(data))
    finally:
        ole.close()
    return f.read_bytes()


def test_doc_encrypted_flag_and_old_word_version_rejected(tmp_path: Path):
    def set_flag(d): d[0x0A:0x0C] = struct.pack("<H", struct.unpack_from("<H", d, 0x0A)[0] | 0x0100)
    with pytest.raises(ParseError, match="암호"):
        extract_text(Path("x.doc"), _patched_ole(tmp_path, doc_bytes(), "WordDocument", set_flag))
    def old(d): d[2:4] = struct.pack("<H", 0x00B0)
    with pytest.raises(ParseError, match="버전"):
        extract_text(Path("x.doc"), _patched_ole(tmp_path, doc_bytes(), "WordDocument", old))
    def bad_sig(d): d[0:2] = b"\x00\x00"
    with pytest.raises(ParseError, match="시그니처"):
        extract_text(Path("x.doc"), _patched_ole(tmp_path, doc_bytes(), "WordDocument", bad_sig))


def test_doc_field_codes_keep_only_the_result():
    s = "링크 \x13 HYPERLINK \"http://x\" \x14보이는 글자\x15 끝\x07셀2\x07\x07다음"
    assert dp._doc_clean(s) == "링크 보이는 글자 끝 | 셀2\n다음"


# ---- .ppt ------------------------------------------------------------------------------------------------------------
def ppt_bytes() -> bytes:
    return F.load(F.FX_ORDER_PPT)


def test_ppt_display_order_notes_tables_and_no_field_placeholders():
    out = extract_text(Path("fx.ppt"), ppt_bytes())
    assert out.splitlines() == ["[슬라이드 1]", "마무리", "문의는 헬프데스크로 연락하세요", "[발표자 노트]", "마지막 노트",
                                "[슬라이드 2]", "개요", "서버 백업 정책을 설명합니다", "[발표자 노트]", "첫 장 발표 노트",
                                "[슬라이드 3]", "일정", "휴가 신청은 팀장 승인이 필요합니다", "구분", "값", "신청", "전날 18시"]
    assert "*" not in out


def test_ppt_encrypted_and_masquerading(tmp_path: Path):
    def enc(d): d[12:16] = struct.pack("<I", 0xF3D1C4DF)
    with pytest.raises(ParseError, match="암호"):
        extract_text(Path("x.ppt"), _patched_ole(tmp_path, ppt_bytes(), "Current User", enc))
    pptx = pytest.importorskip("pptx")
    p = pptx.Presentation(); s = p.slides.add_slide(p.slide_layouts[1]); s.shapes.title.text = "pptx 가짜 ppt"; b = io.BytesIO(); p.save(b)
    assert "pptx 가짜 ppt" in extract_text(Path("x.ppt"), b.getvalue())
    with pytest.raises(ParseError):
        extract_text(Path("x.ppt"), b"nope")


def test_ppt_falls_back_to_file_order_when_directory_is_broken(tmp_path: Path):
    def smash(d): d[16:20] = struct.pack("<I", 0xFFFFFF00)     # offsetToCurrentEdit 를 엉뚱한 곳으로
    out = extract_text(Path("x.ppt"), _patched_ole(tmp_path, ppt_bytes(), "Current User", smash))
    assert out.count("[슬라이드") == 3 and "개요" in out and "마무리" in out


# ---- 손상 파일 퍼징 (실제 바이너리 기반) -------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,raw,suffix", [("doc", doc_bytes(), ".doc"), ("ppt", ppt_bytes(), ".ppt")])
def test_fuzz_corrupted_binaries_never_hang_or_leak_other_exceptions(name, raw, suffix):
    rng = random.Random(11)
    worst = 0.0
    outcomes = {"ok": 0, "parse_error": 0}
    for _ in range(150):
        data = bytearray(raw)
        mode = rng.choice(["flip", "truncate", "zero", "garbage"])
        if mode == "flip":
            for _ in range(rng.randint(1, 30)):
                data[rng.randrange(len(data))] ^= 1 << rng.randrange(8)
        elif mode == "truncate":
            del data[rng.randrange(1, len(data)):]
        elif mode == "zero":
            a = rng.randrange(len(data)); data[a : a + rng.randint(1, 4096)] = b"\x00" * min(4096, len(data) - a)
        else:
            a = rng.randrange(len(data)); data[a : a + 64] = bytes(rng.randrange(256) for _ in range(min(64, len(data) - a)))
        t = time.perf_counter()
        try:
            extract_text(Path("f" + suffix), bytes(data)); outcomes["ok"] += 1
        except ParseError:
            outcomes["parse_error"] += 1
        worst = max(worst, time.perf_counter() - t)
    assert worst < 3.0 and sum(outcomes.values()) == 150


def test_loader_reads_new_formats_end_to_end(tmp_path: Path):
    root = tmp_path / "docs"; root.mkdir()
    (root / "a.doc").write_bytes(doc_bytes()); (root / "b.ppt").write_bytes(ppt_bytes())
    (root / "c.odt").write_bytes(odf("text", "<text:p>서버 백업 지침</text:p>")); (root / "d.ods").write_bytes(odf("spreadsheet", '<table:table table:name="s"><table:table-row><table:table-cell><text:p>x</text:p></table:table-cell></table:table-row></table:table>'))
    (root / "e.odp").write_bytes(odf("presentation", "<draw:page><draw:frame><draw:text-box><text:p>슬라이드 본문</text:p></draw:text-box></draw:frame></draw:page>"))
    loader = SafeDocumentLoader(root, (".doc", ".ppt", ".odt", ".ods", ".odp"), 10_000_000)
    docs = {p.name: loader.load(p) for p in loader.discover()}
    assert set(docs) == {"a.doc", "b.ppt", "c.odt", "d.ods", "e.odp"}
    assert all(d.text.strip() and len(d.sha256) == 64 for d in docs.values())
    assert "서버 운영 지침" in docs["a.doc"].text and "마무리" in docs["b.ppt"].text


def test_ods_repeated_value_rows_are_capped_not_exploded():
    body = ('<table:table table:name="s"><table:table-row table:number-rows-repeated="500"><table:table-cell><text:p>x</text:p></table:table-cell></table:table-row></table:table>')
    lines = extract_text(Path("a.ods"), odf("spreadsheet", body)).splitlines()
    assert lines == ["[시트 s]"] + ["x"] * dp.MAX_ODS_REPEAT


def test_odt_table_empty_repeated_cells_collapse_to_one_slot():
    body = ('<table:table><table:table-row><table:table-cell><text:p>a</text:p></table:table-cell>'
            '<table:table-cell table:number-columns-repeated="8"/><table:table-cell><text:p>b</text:p></table:table-cell></table:table-row></table:table>')
    assert "a |  | b" in extract_text(Path("a.odt"), odf("text", body))

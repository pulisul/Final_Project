import io
import random
import time
import zipfile
from pathlib import Path

import pytest

from intellectagent import doc_parsers as dp
from intellectagent.doc_parsers import ParseError, extract_text
from intellectagent.hashing import sha256_bytes
from intellectagent.loaders import SafeDocumentLoader, UnsupportedFileError

from .hwp_fixture import make_hwp

ALL = (".md", ".txt", ".pdf", ".docx", ".xlsx", ".pptx", ".hwpx", ".hwp")


def make_hwpx(sections: dict, encrypted=False) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, paras in sections.items():
            ps = "".join(f"<hp:p><hp:run><hp:t>{p}</hp:t></hp:run></hp:p>" for p in paras)
            z.writestr(name, f'<?xml version="1.0"?><hs:sec xmlns:hs="s" xmlns:hp="p">{ps}</hs:sec>')
        if encrypted:
            z.writestr("META-INF/manifest.xml", "<m><encryption-data/></m>")
    return buf.getvalue()


def make_docx(tmp_path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("휴가 규정", 1)
    d.add_paragraph("연차는 입사 1년 후부터 15일이 부여된다.")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text, t.cell(1, 0).text, t.cell(1, 1).text = "구분", "일수", "연차", "15"
    p = tmp_path / "a.docx"; d.save(p); return p


def make_xlsx(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "비용"
    ws.append(["항목", "금액"]); ws.append(["교통비", 12000]); ws.append([None, True])
    p = tmp_path / "a.xlsx"; wb.save(p); return p


def make_pptx(tmp_path, reorder=False, footer=False):  # footer: False | "ph"(슬라이드 번호 자리표시자) | "plain"(일반 텍스트 상자)
    pptx = pytest.importorskip("pptx")
    from pptx.util import Inches
    prs = pptx.Presentation()
    layout = prs.slide_layouts[5]
    for i, title in enumerate(["첫째 슬라이드", "둘째 슬라이드", "셋째 슬라이드"], 1):
        s = prs.slides.add_slide(layout)
        s.shapes.title.text = title
        if i == 2:
            tbl = s.shapes.add_table(2, 2, Inches(1), Inches(2), Inches(4), Inches(1)).table
            tbl.cell(0, 0).text, tbl.cell(0, 1).text, tbl.cell(1, 0).text, tbl.cell(1, 1).text = "항목", "값", "백업", "30일"
            s.notes_slide.notes_text_frame.text = "발표자 메모: 보관 기간 강조"
        if footer and i == 1:
            tb = s.shapes.add_textbox(Inches(1), Inches(6), Inches(1), Inches(0.5))
            tb.text_frame.text = "‹#›"
            if footer == "ph":
                from lxml import etree
                ph = etree.SubElement(tb._element.nvSpPr.nvPr, "{http://schemas.openxmlformats.org/presentationml/2006/main}ph"); ph.set("type", "sldNum")
    if reorder:  # 셋째 슬라이드를 맨 앞으로: 파일 번호(slide3.xml)와 표시 순서가 달라진다
        lst = prs.slides._sldIdLst; items = list(lst); lst.remove(items[2]); lst.insert(0, items[2])
    p = tmp_path / "a.pptx"; prs.save(p); return p


# ---- 형식별 추출 ------------------------------------------------------------------
def test_docx_text_and_table(tmp_path):
    t = extract_text(make_docx(tmp_path))
    assert "휴가 규정" in t and "연차는 입사 1년 후부터" in t and "구분 | 일수" in t and "연차 | 15" in t


def test_xlsx_sheet_rows(tmp_path):
    t = extract_text(make_xlsx(tmp_path))
    assert "비용" in t and "교통비" in t and "12000" in t


def test_pptx_text_table_notes(tmp_path):
    t = extract_text(make_pptx(tmp_path))
    assert "[슬라이드 1]" in t and "첫째 슬라이드" in t
    assert "항목 | 값" in t and "백업 | 30일" in t
    assert "[발표자 노트]" in t and "보관 기간 강조" in t
    assert t.index("첫째") < t.index("둘째") < t.index("셋째")


def test_pptx_follows_display_order_not_file_number(tmp_path):
    t = extract_text(make_pptx(tmp_path, reorder=True))
    assert t.index("셋째") < t.index("첫째") < t.index("둘째")  # 표시 순서
    assert t.index("[슬라이드 1]") < t.index("셋째")


def test_pptx_skips_slide_number_placeholder(tmp_path):
    assert "‹#›" not in extract_text(make_pptx(tmp_path, footer="ph"))
    assert "‹#›" in extract_text(make_pptx(tmp_path, footer="plain"))  # 대조군: 일반 텍스트 상자의 같은 글자는 내용으로 읽힌다


def test_hwpx_sections_numeric_order():
    data = make_hwpx({"Contents/section10.xml": ["열번째"], "Contents/section2.xml": ["두번째"]})
    t = dp._PARSERS[".hwpx"](data, Path("a.hwpx"))
    assert t.index("두번째") < t.index("열번째")


def test_hwp_roundtrip_and_failure_reasons(tmp_path):
    p = tmp_path / "a.hwp"; p.write_bytes(make_hwp([["안녕하세요", "한글 문서"], ["둘째 구역"]]))
    t = extract_text(p)
    assert "안녕하세요" in t and "둘째 구역" in t
    p.write_bytes(make_hwp([["x"]], encrypted=True))
    with pytest.raises(ParseError):
        extract_text(p)


REAL_HWP = Path(__file__).resolve().parents[1] / "fixtures" / "hwp5" / "minimal-body-table.hwp"
REAL_HWP_SHA256 = "12110af5fb8697c90942d001ba5306e8fcf54f6655f453cb14f8436654fbee2f"


def test_real_hancom_hwp_file_is_read_through_parser_and_loader():
    """합성 파일이 아닌 한컴이 저장한 실제 .hwp (표 포함). 파일이 바뀌지 않았는지 해시로 먼저 확인한다."""
    pytest.importorskip("olefile")
    assert sha256_bytes(REAL_HWP.read_bytes()) == REAL_HWP_SHA256
    t = extract_text(REAL_HWP)
    for marker in ("편집 탭 – 표", "제목", "담당자", "세부 내용", "표 붙이기"):
        assert marker in t
    assert "\x00" not in t and "\ufffd" not in t            # 제어문자·깨진 글자가 섞이지 않는다
    ld = SafeDocumentLoader(REAL_HWP.parent, ALL, 10_000_000)
    assert REAL_HWP in ld.discover()
    doc = ld.load(REAL_HWP)
    assert "편집 탭 – 표" in doc.text and doc.metadata["suffix"] == ".hwp"


# ---- 보안·견고성 ---------------------------------------------------------------------
def test_xml_entity_bomb_rejected(tmp_path):
    p = make_pptx(tmp_path)
    buf = io.BytesIO()
    with zipfile.ZipFile(p) as zin, zipfile.ZipFile(buf, "w") as zout:
        for n in zin.namelist():
            raw = zin.read(n)
            if n == "ppt/slides/slide1.xml":
                raw = b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]>' + raw.split(b"?>", 1)[1]
            zout.writestr(n, raw)
    q = tmp_path / "bomb.pptx"; q.write_bytes(buf.getvalue())
    with pytest.raises(ParseError):
        extract_text(q)


def test_zip_bomb_rejected_by_size_limit(tmp_path, monkeypatch):
    p = make_pptx(tmp_path)
    monkeypatch.setattr(dp, "MAX_UNZIPPED_BYTES", 1000)
    with pytest.raises(ParseError):
        extract_text(p)


def test_encrypted_office_container_rejected(tmp_path):
    for ext in (".docx", ".xlsx", ".pptx"):
        q = tmp_path / f"enc{ext}"; q.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 600)
        with pytest.raises(ParseError):
            extract_text(q)


def test_fuzz_corrupted_files_never_crash_or_hang(tmp_path):
    seeds = [make_docx(tmp_path), make_xlsx(tmp_path), make_pptx(tmp_path)]
    seeds.append(tmp_path / "s.hwpx"); seeds[-1].write_bytes(make_hwpx({"Contents/section0.xml": ["가나다"]}))
    rng = random.Random(7); t0 = time.time(); ok = bad = 0
    for _ in range(300):
        src = rng.choice(seeds); b = bytearray(src.read_bytes())
        for _ in range(rng.randint(1, 20)):
            b[rng.randrange(len(b))] = rng.randrange(256)
        if rng.random() < 0.3:
            b = b[: rng.randrange(1, len(b))]
        q = tmp_path / f"f{src.suffix}"; q.write_bytes(bytes(b))
        try:
            extract_text(q); ok += 1
        except ParseError:
            bad += 1
    assert ok + bad == 300 and time.time() - t0 < 60


def test_unsupported_extension():
    with pytest.raises(ParseError):
        extract_text(Path("a.doc"), b"x")


# ---- 로더 통합 -----------------------------------------------------------------------
def test_loader_reads_office_formats_and_hashes_the_bytes_it_parsed(tmp_path):
    for mk in (make_docx, make_xlsx, make_pptx):
        mk(tmp_path)
    (tmp_path / "a.hwpx").write_bytes(make_hwpx({"Contents/section0.xml": ["한글 문서"]}))
    ld = SafeDocumentLoader(tmp_path, ALL, 10_000_000)
    names = {p.name for p in ld.discover()}
    assert {"a.docx", "a.xlsx", "a.pptx", "a.hwpx"} <= names
    for n in ("a.docx", "a.xlsx", "a.pptx", "a.hwpx"):
        doc = ld.load(tmp_path / n)
        assert doc.text.strip() and doc.sha256 == sha256_bytes((tmp_path / n).read_bytes()) and doc.metadata["suffix"] == Path(n).suffix


def test_loader_keeps_default_extension_policy(tmp_path):
    p = make_docx(tmp_path)
    with pytest.raises(UnsupportedFileError):
        SafeDocumentLoader(tmp_path, (".md", ".txt", ".pdf"), 10_000_000).load(p)  # 설정에서 허용하지 않으면 여전히 거부


def test_loader_cp949_text_is_not_garbled(tmp_path):
    p = tmp_path / "memo.txt"; p.write_bytes("연차 휴가는 15일입니다".encode("cp949"))
    assert SafeDocumentLoader(tmp_path, ALL, 10_000_000).load(p).text == "연차 휴가는 15일입니다"


def test_loader_rejects_empty_and_corrupt_office_files(tmp_path):
    ld = SafeDocumentLoader(tmp_path, ALL, 10_000_000)
    (tmp_path / "bad.docx").write_bytes(b"not a zip")
    with pytest.raises(UnsupportedFileError):
        ld.load(tmp_path / "bad.docx")


# ---- Streamlit 업로드 경로(save_upload_bytes) 통합 ----------------------------------------
def test_upload_path_accepts_office_and_removes_corrupt_files(tmp_path):
    from intellectagent.document_ingest import UploadValidationError, save_upload_bytes
    src = make_pptx(tmp_path)
    root = tmp_path / "real_docs"
    ok = save_upload_bytes(allowed_root=root, filename="회의록.pptx", data=src.read_bytes(),
                           allowed_extensions=ALL, max_file_bytes=10_000_000)
    assert Path(ok.path).exists() and ok.stored_name == "회의록.pptx"
    with pytest.raises(Exception):  # 손상 파일: 로더의 2차 검증에서 거부되고 디스크에도 남지 않는다
        save_upload_bytes(allowed_root=root, filename="bad.docx", data=b"PK\x03\x04garbage",
                          allowed_extensions=ALL, max_file_bytes=10_000_000)
    assert not (root / "bad.docx").exists()

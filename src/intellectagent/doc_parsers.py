"""
doc_parsers.py — 문서 형식별 텍스트 추출.

IntellectAgent_AI/src/parsers.py(자체 구현, 테스트·퍼징 통과)를 이 저장소로 옮긴 것이며 .pptx 를 추가했다.
AutoRAG 2.0 이 .hwp/.hwpx/.docx/.xlsx/.pdf 는 kordoc, .pptx 는 자체 리더로 읽는 구성에서 착안했다.
이 모듈은 외부 서비스·Node 없이 순수 파이썬(+선택적 pypdf/olefile)으로 동작한다.

지원: .md .txt (UTF-8, 실패 시 CP949) / .pdf (pypdf) / .docx / .xlsx / .pptx / .hwpx / .hwp(5.x)
 - docx/xlsx/hwpx 는 ZIP+XML 이므로 표준 라이브러리만으로 읽는다 (추가 설치 없음).
 - hwp 는 OLE 컨테이너라 olefile 이 필요하다 (pip install olefile). 없으면 '왜 못 읽었는지'가 사유로 남는다.

원칙
 - 못 읽는 파일은 조용히 빈 문자열을 돌려주지 않고 ParseError 로 이유를 알린다
   (암호화·배포용 hwp, 비밀번호 걸린 docx/xlsx, 손상 파일, 압축폭탄 등). 호출자는 이를 index_skips 에 기록한다.
 - 보안: DOCTYPE/ENTITY 가 있는 XML 은 거부(엔티티 폭주 방지), ZIP 은 압축 해제 크기 상한, 출력 글자수 상한.
 - 내용 바이트를 호출자가 이미 읽었다면 data= 로 넘긴다 (색인에 쓴 바이트와 해시 대상이 일치해야 하므로 파일을 두 번 읽지 않는다).
"""

from __future__ import annotations

import io
import re
import struct
import unicodedata
import zipfile
import zlib
from pathlib import Path
from xml.etree import ElementTree as ET

MAX_UNZIPPED_BYTES = 200 * 1024 * 1024   # 압축 해제 총량 상한 (압축폭탄 방어)
MAX_PART_BYTES = 80 * 1024 * 1024        # XML 한 조각 상한
MAX_TEXT_CHARS = 5_000_000               # 추출 텍스트 상한 (거대한 시트 등)


class ParseError(RuntimeError):
    """형식은 지원하지만 이 파일은 읽을 수 없음 (사유를 메시지에 담는다)."""


# ---------------------------------------------------------------------------
# 공통 도구
# ---------------------------------------------------------------------------
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _open_zip(data: bytes) -> zipfile.ZipFile:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ParseError("손상된 파일이거나 ZIP 형식이 아님") from exc
    if sum(i.file_size for i in zf.infolist()) > MAX_UNZIPPED_BYTES:
        raise ParseError("압축 해제 크기가 상한을 넘음 (압축폭탄 의심)")
    return zf


def _read_part(zf: zipfile.ZipFile, name: str) -> bytes:
    info = zf.getinfo(name)
    if info.file_size > MAX_PART_BYTES:
        raise ParseError(f"내부 조각이 너무 큼: {name}")
    with zf.open(info) as f:
        raw = f.read(MAX_PART_BYTES + 1)
    if len(raw) > MAX_PART_BYTES:
        raise ParseError(f"내부 조각이 너무 큼: {name}")
    return raw


def _parse_xml(raw: bytes, name: str) -> ET.Element:
    head = raw[:4096].upper()
    if b"<!DOCTYPE" in head or b"<!ENTITY" in raw[:65536].upper():
        raise ParseError(f"XML 에 DOCTYPE/ENTITY 선언이 있어 거부함: {name}")
    try:
        return ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ParseError(f"XML 해석 실패: {name} ({exc})") from exc


def _clip(text: str) -> str:
    return text[:MAX_TEXT_CHARS]


def _is_encrypted_ooxml(data: bytes) -> bool:
    # 암호 걸린 Office 파일은 ZIP 이 아니라 OLE(D0CF11E0) 로 저장된다
    return data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


# ---------------------------------------------------------------------------
# txt / md / pdf
# ---------------------------------------------------------------------------
def _decode_text(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):      # 윈도우 메모장 '유니코드' 저장(UTF-16 + BOM)
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            pass
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp949")           # 한국어 윈도우에서 저장한 메모장 파일
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="ignore")


def _parse_text(data: bytes, path: Path) -> str:
    return _decode_text(data)


_LIGATURES = {chr(c): unicodedata.normalize("NFKC", chr(c)) for c in range(0xFB00, 0xFB07)}  # ﬁ ﬂ ﬀ ﬃ ﬄ ﬅ ﬆ -> fi fl ff ffi ffl st st


def clean_pdf_page(page: str) -> str:
    """pypdf 가 내놓은 한 페이지 텍스트를 검색에 쓰기 좋게 정리한다.
    - 합자(ﬁ 등)를 풀어 쓴다: 'ofﬁcial' 이 'official' 질의에 걸리지 않는 문제를 막는다.
    - 줄 끝 공백 제거, NUL 제거.
    - 낱말마다 줄이 바뀌고 그 사이에 빈 줄이 끼는 추출(글자를 위치로 찍은 PDF)은 한 줄로 이어 붙인다.
      (이대로 두면 청커가 빈 줄을 문단 경계로 보고 낱말 하나하나를 청크로 만든다.) 3줄 이상 연속 개행은 문단 경계로 남긴다.
    일반 PDF(줄 평균 길이가 충분히 긴 경우)는 위 첫 두 가지만 적용되어 내용이 달라지지 않는다."""
    page = page.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    for k, v in _LIGATURES.items():
        page = page.replace(k, v)
    lines = [ln.rstrip() for ln in page.split("\n")]
    page = "\n".join(lines)
    nonempty = [ln for ln in lines if ln]
    if len(nonempty) >= 8 and sum(map(len, nonempty)) / len(nonempty) < 20:
        pairs = [(a, b) for a, b in zip(lines, lines[1:])]
        # 낱말-빈줄-낱말 패턴이 우세한지: 비어 있지 않은 줄 다음 줄이 빈 줄인 비율
        blank_after = sum(1 for a, b in pairs if a and not b) / max(1, len(nonempty))
        if blank_after >= 0.6:
            page = re.sub(r"\n{3,}", "\x00", page)
            page = re.sub(r"\s*\n\s*", " ", page).replace("\x00", "\n\n")
    return page.strip()



def _parse_pdf(data: bytes, path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ParseError("pypdf 미설치 — pip install pypdf") from exc
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                ok = reader.decrypt("")
            except Exception:  # noqa: BLE001
                ok = 0
            if not ok:
                raise ParseError("암호가 걸린 PDF")
        return _clip("\n\n".join(clean_pdf_page(page.extract_text() or "") for page in reader.pages))
    except ParseError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"PDF 읽기 실패: {type(exc).__name__}: {exc}") from exc


# ---------------------------------------------------------------------------
# docx
# ---------------------------------------------------------------------------
def _docx_paragraph_text(p: ET.Element) -> str:
    out: list[str] = []
    for el in p.iter():
        name = _local(el.tag)
        if name == "t" and el.text:
            out.append(el.text)
        elif name == "tab":
            out.append("\t")
        elif name in ("br", "cr"):
            out.append("\n")
    return "".join(out)


def _docx_walk(el: ET.Element, lines: list[str]) -> None:
    for child in el:
        name = _local(child.tag)
        if name == "p":
            lines.append(_docx_paragraph_text(child))
        elif name == "tbl":
            for tr in child.iter():
                if _local(tr.tag) != "tr":
                    continue
                cells = []
                for tc in tr:
                    if _local(tc.tag) != "tc":
                        continue
                    sub: list[str] = []
                    _docx_walk(tc, sub)
                    cells.append(" ".join(s.strip() for s in sub if s.strip()))
                if any(cells):
                    lines.append(" | ".join(cells))
        elif name in ("body", "sdt", "sdtContent", "customXml", "ins", "smartTag"):
            _docx_walk(child, lines)


def _parse_docx(data: bytes, path: Path) -> str:
    if _is_encrypted_ooxml(data):
        raise ParseError("암호가 걸린 Word 문서")
    zf = _open_zip(data)
    names = set(zf.namelist())
    if "word/document.xml" not in names:
        raise ParseError("word/document.xml 이 없음 (docx 가 아니거나 손상)")
    lines: list[str] = []
    _docx_walk(_parse_xml(_read_part(zf, "word/document.xml"), "word/document.xml"), lines)
    for extra in sorted(n for n in names if re.fullmatch(r"word/(footnotes|endnotes)\.xml", n)):
        sub: list[str] = []
        _docx_walk(_parse_xml(_read_part(zf, extra), extra), sub)
        lines.extend(s for s in sub if s.strip())
    # 빈 줄이 연속되면 한 줄로
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return _clip(text)


# ---------------------------------------------------------------------------
# xlsx
# ---------------------------------------------------------------------------
def _col_index(ref: str) -> int:
    letters = re.match(r"[A-Za-z]+", ref or "")
    n = 0
    for ch in (letters.group(0).upper() if letters else ""):
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def _xlsx_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = _parse_xml(_read_part(zf, "xl/sharedStrings.xml"), "xl/sharedStrings.xml")
    out = []
    for si in root:
        if _local(si.tag) == "si":
            out.append("".join(t.text or "" for t in si.iter() if _local(t.tag) == "t"))
    return out


def _xlsx_sheet_list(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    """(시트 이름, 파트 경로) 목록 — workbook.xml + rels 로 정확히 연결, 실패하면 파일명 순서로 폴백."""
    names = zf.namelist()
    try:
        wb = _parse_xml(_read_part(zf, "xl/workbook.xml"), "xl/workbook.xml")
        rels = _parse_xml(_read_part(zf, "xl/_rels/workbook.xml.rels"), "workbook.xml.rels")
        target = {r.get("Id"): r.get("Target") for r in rels}
        result = []
        for sh in wb.iter():
            if _local(sh.tag) != "sheet":
                continue
            rid = next((v for k, v in sh.attrib.items() if _local(k) == "id"), None)
            tgt = target.get(rid, "")
            if not tgt:
                continue
            part = tgt.lstrip("/") if tgt.startswith("/") else "xl/" + tgt
            if part in names:
                result.append((sh.get("name") or part, part))
        if result:
            return result
    except (KeyError, ParseError):
        pass
    parts = sorted(n for n in names if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))
    return [(Path(p).stem, p) for p in parts]


def _parse_xlsx(data: bytes, path: Path) -> str:
    if _is_encrypted_ooxml(data):
        raise ParseError("암호가 걸린 Excel 문서")
    zf = _open_zip(data)
    if "xl/workbook.xml" not in zf.namelist():
        raise ParseError("xl/workbook.xml 이 없음 (xlsx 가 아니거나 손상)")
    shared = _xlsx_shared_strings(zf)
    out: list[str] = []
    total = 0
    for sheet_name, part in _xlsx_sheet_list(zf):
        root = _parse_xml(_read_part(zf, part), part)
        out.append(f"## 시트: {sheet_name}")
        for row in root.iter():
            if _local(row.tag) != "row":
                continue
            cells: dict[int, str] = {}
            for c in row:
                if _local(c.tag) != "c":
                    continue
                ctype = c.get("t")
                val = None
                if ctype == "inlineStr":
                    val = "".join(t.text or "" for t in c.iter() if _local(t.tag) == "t")
                else:
                    v = next((x for x in c if _local(x.tag) == "v"), None)
                    if v is not None and v.text is not None:
                        if ctype == "s":
                            try:
                                val = shared[int(v.text)]
                            except (ValueError, IndexError):
                                val = None
                        elif ctype == "b":
                            val = "TRUE" if v.text.strip() == "1" else "FALSE"
                        else:
                            val = v.text
                if val is not None and val.strip():
                    cells[_col_index(c.get("r", ""))] = val.strip()
            if cells:
                line = " | ".join(cells[k] for k in sorted(cells))
                out.append(line)
                total += len(line)
                if total > MAX_TEXT_CHARS:
                    out.append("(이하 생략: 텍스트 상한 초과)")
                    return _clip("\n".join(out))
    return _clip("\n".join(out))


# ---------------------------------------------------------------------------
# hwpx (한글 2014+ 의 XML 형식)
# ---------------------------------------------------------------------------
def _hwpx_paragraphs(root: ET.Element) -> list[str]:
    lines: list[str] = []

    def walk(el: ET.Element) -> None:
        for child in el:
            name = _local(child.tag)
            if name == "p":
                buf: list[str] = []
                for sub in child.iter():
                    sn = _local(sub.tag)
                    if sn == "t" and sub.text:
                        buf.append(sub.text)
                    elif sn == "tab":
                        buf.append("\t")
                    elif sn == "lineBreak":
                        buf.append("\n")
                if buf:
                    lines.append("".join(buf))
            else:
                walk(child)

    walk(root)
    return lines


def _parse_hwpx(data: bytes, path: Path) -> str:
    zf = _open_zip(data)
    names = zf.namelist()
    if "META-INF/manifest.xml" in names:
        man = _read_part(zf, "META-INF/manifest.xml")
        if b"encryption-data" in man or b"EncryptionData" in man:
            raise ParseError("암호가 걸린 HWPX 문서")
    sections = sorted(
        (n for n in names if re.fullmatch(r"Contents/section\d+\.xml", n)),
        key=lambda n: int(re.findall(r"\d+", n)[-1]),
    )
    if not sections:
        raise ParseError("Contents/section*.xml 이 없음 (hwpx 가 아니거나 손상)")
    lines: list[str] = []
    for n in sections:
        lines.extend(_hwpx_paragraphs(_parse_xml(_read_part(zf, n), n)))
    return _clip("\n".join(lines))


# ---------------------------------------------------------------------------
# hwp 5.x (OLE 컨테이너 + zlib + 레코드)
# ---------------------------------------------------------------------------
HWPTAG_PARA_TEXT = 0x10 + 51  # 67
_SINGLE_CTRL = {0, 10, 13} | set(range(24, 32))   # 1 글자짜리 제어문자
MAX_HWP_STREAM = 200 * 1024 * 1024


def _hwp_decompress(raw: bytes) -> bytes:
    d = zlib.decompressobj(-15)                      # 한글은 헤더 없는 raw deflate 를 쓴다
    out = d.decompress(raw, MAX_HWP_STREAM)
    if d.unconsumed_tail:
        raise ParseError("HWP 본문 압축 해제 크기가 상한을 넘음")
    if not d.eof:
        raise ParseError("HWP 본문 압축 스트림이 끝나지 않음 (파일이 잘렸거나 손상)")
    return out


def _hwp_para_text(payload: bytes) -> str:
    chars: list[str] = []
    i, n = 0, len(payload) - len(payload) % 2
    while i < n:
        code = struct.unpack_from("<H", payload, i)[0]
        if code >= 32:
            chars.append(chr(code))
            i += 2
        elif code in _SINGLE_CTRL:
            if code in (10, 13):
                chars.append("\n")
            elif code in (30, 31):
                chars.append(" ")
            i += 2
        else:
            # 인라인/확장 제어: 코드 + 14바이트 부가정보 = 총 16바이트 (탭(9)은 공백으로 보존)
            if code == 9:
                chars.append("\t")
            i += 16
    return "".join(chars)


def hwp_records(stream: bytes):
    """HWP 레코드 스트림을 (tag_id, level, payload) 로 순회. 잘린/손상 스트림이면 ParseError."""
    i, n = 0, len(stream)
    while i < n:
        if i + 4 > n:
            if not any(stream[i:]):
                return                       # 끝의 0 패딩은 정상으로 본다
            raise ParseError("HWP 레코드 헤더가 잘림 (손상)")
        header = struct.unpack_from("<I", stream, i)[0]
        i += 4
        tag, level, size = header & 0x3FF, (header >> 10) & 0x3FF, header >> 20
        if size == 0xFFF:
            if i + 4 > n:
                raise ParseError("HWP 레코드 크기 필드가 잘림 (손상)")
            size = struct.unpack_from("<I", stream, i)[0]
            i += 4
        if i + size > n:
            raise ParseError("HWP 레코드 본문이 잘림 (손상)")
        yield tag, level, stream[i : i + size]
        i += size


def _parse_hwp(data: bytes, path: Path) -> str:
    try:
        import olefile
    except ImportError as exc:
        raise ParseError("olefile 미설치 — pip install olefile (.hwp 읽기에 필요)") from exc
    try:
        ole = olefile.OleFileIO(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"HWP(OLE) 열기 실패: {type(exc).__name__}: {exc}") from exc
    try:
        if not ole.exists("FileHeader"):
            raise ParseError("FileHeader 가 없음 (HWP 5.x 가 아니거나 손상)")
        header = ole.openstream("FileHeader").read()
        if not header.startswith(b"HWP Document File"):
            raise ParseError("HWP 시그니처 불일치")
        flags = struct.unpack_from("<I", header, 36)[0] if len(header) >= 40 else 0
        compressed, encrypted, distribution = bool(flags & 1), bool(flags & 2), bool(flags & 4)
        if encrypted:
            raise ParseError("암호가 걸린 HWP 문서")
        sections = sorted(
            ("/".join(s) for s in ole.listdir() if len(s) == 2 and s[0] == "BodyText" and s[1].startswith("Section")),
            key=lambda s: int(re.findall(r"\d+", s)[-1]) if re.findall(r"\d+", s) else 0,
        )
        if distribution:
            # 배포용 문서는 본문(ViewText)이 AES 로 암호화되어 있어 읽지 않는다 (복호화 시도 안 함)
            raise ParseError("배포용(열람 제한) HWP 문서 — 본문이 암호화되어 있음")
        if not sections:
            raise ParseError("BodyText/Section 스트림이 없음")
        lines: list[str] = []
        for s in sections:
            raw = ole.openstream(s).read()
            body = _hwp_decompress(raw) if compressed else raw
            for tag, _lvl, payload in hwp_records(body):
                if tag == HWPTAG_PARA_TEXT:
                    t = _hwp_para_text(payload).strip("\n")
                    if t.strip():
                        lines.append(t)
        return _clip("\n".join(lines))
    except ParseError:
        raise
    except zlib.error as exc:
        raise ParseError(f"HWP 본문 압축 해제 실패: {exc}") from exc
    finally:
        ole.close()


# ---------------------------------------------------------------------------
# pptx  (AutoRAG 2.0 의 PptxParser 는 슬라이드·노트 XML 의 텍스트를 모은다. 여기서는 표시 순서·표·노트 연결까지 처리한다)
# ---------------------------------------------------------------------------
_PPTX_SKIP_PLACEHOLDERS = {"sldNum", "hdr", "ftr", "dt"}  # 슬라이드 번호·머리글·바닥글·날짜는 내용이 아니다


def _drawing_paragraph(p: ET.Element) -> str:
    out: list[str] = []
    for el in p.iter():
        name = _local(el.tag)
        if name == "t" and el.text:
            out.append(el.text)
        elif name == "br":
            out.append("\n")
        elif name == "tab":
            out.append("\t")
    return "".join(out)


def _pptx_collect(el: ET.Element, lines: list[str]) -> None:
    for child in el:
        name = _local(child.tag)
        if name == "sp":
            ph = next((x for x in child.iter() if _local(x.tag) == "ph"), None)
            if ph is not None and ph.get("type") in _PPTX_SKIP_PLACEHOLDERS:
                continue
            for p in child.iter():
                if _local(p.tag) == "p":
                    t = _drawing_paragraph(p)
                    if t.strip():
                        lines.append(t)
        elif name == "graphicFrame":  # 표
            for tr in child.iter():
                if _local(tr.tag) != "tr":
                    continue
                cells = []
                for tc in tr:
                    if _local(tc.tag) != "tc":
                        continue
                    cells.append(" ".join(t.strip() for t in (_drawing_paragraph(p) for p in tc.iter() if _local(p.tag) == "p") if t.strip()))
                if any(cells):
                    lines.append(" | ".join(cells))
        elif name == "Fallback":
            continue  # mc:AlternateContent 는 Choice 만 읽는다 (같은 내용이 두 번 들어가는 것을 막는다)
        elif name in ("spTree", "cSld", "grpSp", "AlternateContent", "Choice"):
            _pptx_collect(child, lines)


def _rels(zf: zipfile.ZipFile, part: str) -> dict[str, tuple[str, str]]:
    """part 의 .rels -> {rId: (Type 끝부분, 정규화된 Target 경로)}"""
    d, base = part.rsplit("/", 1)
    rel = f"{d}/_rels/{base}.rels"
    if rel not in zf.namelist():
        return {}
    out: dict[str, tuple[str, str]] = {}
    for r in _parse_xml(_read_part(zf, rel), rel):
        if _local(r.tag) != "Relationship":
            continue
        tgt = r.get("Target", "")
        if r.get("TargetMode") == "External":
            continue
        parts = (d + "/" + tgt).split("/") if not tgt.startswith("/") else tgt.lstrip("/").split("/")
        norm: list[str] = []
        for seg in parts:
            if seg == "..":
                if norm:
                    norm.pop()
            elif seg not in ("", "."):
                norm.append(seg)
        out[r.get("Id", "")] = ((r.get("Type") or "").rsplit("/", 1)[-1], "/".join(norm))
    return out


def _parse_pptx(data: bytes, path: Path) -> str:
    if _is_encrypted_ooxml(data):
        raise ParseError("암호가 걸린 PowerPoint 문서")
    zf = _open_zip(data)
    names = set(zf.namelist())
    slide_re = re.compile(r"ppt/slides/slide\d+\.xml")
    numeric = sorted((n for n in names if slide_re.fullmatch(n)), key=lambda n: int(re.findall(r"\d+", n)[-1]))
    if not numeric:
        raise ParseError("ppt/slides/slide*.xml 이 없음 (pptx 가 아니거나 손상)")
    # 표시 순서: presentation.xml 의 sldIdLst + rels. (슬라이드를 옮기면 파일 번호와 표시 순서가 달라진다)
    ordered: list[str] = []
    if "ppt/presentation.xml" in names:
        rels = _rels(zf, "ppt/presentation.xml")
        pres = _parse_xml(_read_part(zf, "ppt/presentation.xml"), "ppt/presentation.xml")
        for el in pres.iter():
            if _local(el.tag) == "sldId":
                rid = next((v for k, v in el.attrib.items() if k.endswith("}id") and k.startswith("{http://schemas.openxmlformats.org/officeDocument/2006/relationships")), None)
                target = rels.get(rid or "", ("", ""))[1]
                if target in names and target not in ordered:
                    ordered.append(target)
    ordered += [n for n in numeric if n not in ordered]  # sldIdLst 에 없는 조각도 버리지 않는다
    blocks: list[str] = []
    for i, slide in enumerate(ordered, 1):
        lines: list[str] = []
        _pptx_collect(_parse_xml(_read_part(zf, slide), slide), lines)
        notes_lines: list[str] = []
        for typ, target in _rels(zf, slide).values():
            if typ == "notesSlide" and target in names:
                _pptx_collect(_parse_xml(_read_part(zf, target), target), notes_lines)
        parts = [f"[슬라이드 {i}]"] + lines
        if notes_lines:
            parts += ["[발표자 노트]"] + notes_lines
        if len(parts) > 1:
            blocks.append("\n".join(parts))
    return _clip("\n\n".join(blocks))


# ---------------------------------------------------------------------------
# OpenDocument (.odt .ods .odp) — ZIP + content.xml
# ---------------------------------------------------------------------------
_OD_TEXT = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
_OD_TABLE = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
_OD_PRES = "urn:oasis:names:tc:opendocument:xmlns:presentation:1.0"
_OD_SKIP_CLASSES = {"page-number", "header", "footer", "date-time"}
MAX_ODS_REPEAT = 10  # 값이 있는 셀/행의 repeated 속성을 펼치는 상한. 값 없는 반복은 건너뛴다(시트 끝의 100만 행 빈칸 방어)
_OD_CONTAINERS = {"list", "list-item", "list-header", "section", "frame", "text-box", "a", "span", "index-body",
                  "table-of-content", "alphabetical-index", "illustration-index", "table-index", "object-index", "user-index",
                  "bibliography", "custom-shape", "g", "caption"}


def _odq(ns: str, name: str) -> str:
    return f"{{{ns}}}{name}"


def _odf_inline(el: ET.Element) -> str:
    """문단/제목 안의 글자. 공백 반복(text:s)·탭·줄바꿈을 복원하고 각주는 [각주: …] 로 인라인한다. 메모(주석)·책갈피는 버린다."""
    parts: list[str] = [el.text or ""]
    for ch in el:
        tag = _local(ch.tag)
        if tag == "s":
            try:
                parts.append(" " * min(int(ch.get(_odq(_OD_TEXT, "c"), "1")), 1000))
            except ValueError:
                parts.append(" ")
        elif tag == "tab":
            parts.append("\t")
        elif tag == "line-break":
            parts.append("\n")
        elif tag == "note":
            body = next((b for b in ch if _local(b.tag) == "note-body"), None)
            if body is not None:
                note = " ".join(t for t in (_odf_inline(p).strip() for p in body.iter() if _local(p.tag) in ("p", "h")) if t)
                if note:
                    parts.append(f"[각주: {note}]")
        elif tag in ("annotation", "annotation-end", "bookmark", "bookmark-start", "bookmark-end", "page-number", "page-count"):
            pass
        else:
            parts.append(_odf_inline(ch))
        parts.append(ch.tail or "")
    return "".join(parts)


def _odf_rows(tbl: ET.Element):
    for ch in tbl:
        tag = _local(ch.tag)
        if tag == "table-row":
            yield ch
        elif tag in ("table-header-rows", "table-rows", "table-row-group"):
            yield from _odf_rows(ch)


def _odf_repeat(el: ET.Element, attr: str) -> int:
    try:
        return max(1, int(el.get(_odq(_OD_TABLE, attr), "1") or 1))
    except ValueError:
        return 1


def _odf_table(tbl: ET.Element, lines: list[str], sheet_mode: bool) -> None:
    for row in _odf_rows(tbl):
        cells: list[str] = []
        for cell in row:
            if _local(cell.tag) not in ("table-cell", "covered-table-cell"):
                continue
            rep = min(_odf_repeat(cell, "number-columns-repeated"), MAX_ODS_REPEAT)
            text = ""
            if _local(cell.tag) == "table-cell":
                paras: list[str] = []
                _odf_blocks(cell, paras)
                text = " ".join(t.strip() for t in paras if t.strip())
            # 값 없는 반복 칸은 위치만 차지한다. 시트는 위치 보존을 위해 펼치되 상한을 둔다.
            cells.extend([text] * rep if text or sheet_mode else [""])
        while cells and not cells[-1]:
            cells.pop()
        if any(cells):
            lines.extend([" | ".join(cells)] * min(_odf_repeat(row, "number-rows-repeated"), MAX_ODS_REPEAT))


def _odf_blocks(el: ET.Element, lines: list[str]) -> None:
    for ch in el:
        tag = _local(ch.tag)
        if tag in ("p", "h"):
            t = _odf_inline(ch).strip()
            if t:
                lines.append(t)
        elif tag == "table" and ch.tag == _odq(_OD_TABLE, "table"):
            _odf_table(ch, lines, sheet_mode=False)
        elif tag in _OD_CONTAINERS:
            if ch.get(_odq(_OD_PRES, "class")) in _OD_SKIP_CLASSES:
                continue  # 슬라이드 번호·머리글·바닥글·날짜 자리는 내용이 아니다 (pptx 파서와 같은 원칙)
            _odf_blocks(ch, lines)


def _parse_odf(data: bytes, path: Path) -> str:
    zf = _open_zip(data)
    names = set(zf.namelist())
    if "META-INF/manifest.xml" in names and b"encryption-data" in _read_part(zf, "META-INF/manifest.xml"):
        raise ParseError("암호가 걸린 OpenDocument 파일")
    if "content.xml" not in names:
        raise ParseError("content.xml 이 없음 (OpenDocument 가 아니거나 손상)")
    root = _parse_xml(_read_part(zf, "content.xml"), "content.xml")
    body = next((b for b in root if _local(b.tag) == "body"), None)
    inner = next((c for c in body if _local(c.tag) in ("text", "spreadsheet", "presentation")), None) if body is not None else None
    if inner is None:
        raise ParseError("본문(office:body)을 찾지 못함")
    lines: list[str] = []
    kind = _local(inner.tag)
    if kind == "spreadsheet":
        for tbl in inner.iter(_odq(_OD_TABLE, "table")):
            rows: list[str] = []
            _odf_table(tbl, rows, sheet_mode=True)
            if rows:
                lines.append(f"[시트 {tbl.get(_odq(_OD_TABLE, 'name'), '')}]")
                lines.extend(rows)
    elif kind == "presentation":
        for n, page in enumerate((p for p in inner if _local(p.tag) == "page"), 1):
            lines.append(f"[슬라이드 {n}]")
            _odf_blocks(page, lines)
            for notes in (c for c in page if _local(c.tag) == "notes"):
                nl: list[str] = []
                _odf_blocks(notes, nl)
                if nl:
                    lines.append("[발표자 노트]")
                    lines.extend(nl)
    else:
        _odf_blocks(inner, lines)
    return _clip("\n".join(lines))


# ---------------------------------------------------------------------------
# .doc (Word 97–2003 바이너리, OLE) — FIB + piece table(CLX)
# ---------------------------------------------------------------------------
def _ole_open(data: bytes, what: str):
    try:
        import olefile
    except ImportError as exc:
        raise ParseError(f"olefile 미설치 — pip install olefile ({what} 읽기에 필요)") from exc
    try:
        return olefile.OleFileIO(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"{what}(OLE) 열기 실패: {type(exc).__name__}: {exc}") from exc


def _doc_clean(s: str) -> str:
    # 필드: \x13 명령 \x14 결과 \x15 — 결과만 남긴다(중첩은 안쪽부터 반복 제거)
    for _ in range(8):
        n = re.sub(r"\x13[^\x13\x14\x15]*\x14", "", s)
        n = re.sub(r"\x13[^\x13\x14\x15]*\x15", "", n)
        if n == s:
            break
        s = n
    s = s.replace("\x07\x07", "\n").replace("\x07", " | ")  # 표: 셀 끝 \x07, 행 끝 \x07\x07
    s = s.replace("\r", "\n").replace("\x0b", "\n").replace("\x0c", "\n").replace("\x1e", "-").replace("\x1f", "")
    s = re.sub(r"[\x00-\x08\x0e-\x1f\x7f]", "", s).replace("￾", "").replace("￿", "")
    return re.sub(r"[ \t]+\n", "\n", s)


def _parse_doc(data: bytes, path: Path) -> str:
    if data[:2] == b"PK":                       # 확장자만 .doc 이고 실제로는 docx
        return _parse_docx(data, path)
    if data[:5] == b"{\\rtf":
        raise ParseError("RTF 형식은 지원하지 않음 (docx 로 저장해 주세요)")
    ole = _ole_open(data, "DOC")
    try:
        if not ole.exists("WordDocument"):
            raise ParseError("WordDocument 스트림이 없음 (Word 문서가 아니거나 손상)")
        wd = ole.openstream("WordDocument").read()
        if len(wd) < 0x1AA or struct.unpack_from("<H", wd, 0)[0] != 0xA5EC:
            raise ParseError("Word FIB 시그니처 불일치 (Word 97 이전 형식이거나 손상)")
        nfib, flags = struct.unpack_from("<H", wd, 2)[0], struct.unpack_from("<H", wd, 0x0A)[0]
        if flags & 0x0100:
            raise ParseError("암호가 걸린 Word 문서")
        if nfib < 0x00C1:
            raise ParseError(f"지원하지 않는 Word 버전 (nFib=0x{nfib:04X})")
        table_name = "1Table" if flags & 0x0200 else "0Table"
        if not ole.exists(table_name):
            raise ParseError(f"{table_name} 스트림이 없음 (손상)")
        table = ole.openstream(table_name).read()
        if struct.unpack_from("<H", wd, 32)[0] != 14 or struct.unpack_from("<H", wd, 62)[0] != 22:
            raise ParseError("FIB 구조가 표준(Word 97–2003)과 다름")
        # FibRgLw97 (오프셋 64): cbMac, lProductCreated, lProductRevised, ccpText, ccpFtn, ccpHdd, (예약), ccpAtn, ccpEdn, ...
        ccp = struct.unpack_from("<9i", wd, 64)
        ccp_text, ccp_ftn, ccp_hdd, ccp_atn, ccp_edn = ccp[3], ccp[4], ccp[5], ccp[7], ccp[8]
        if min(ccp_text, ccp_ftn, ccp_hdd, ccp_atn, ccp_edn) < 0:
            raise ParseError("Word 문자 수 필드가 음수 (손상)")
        fc_clx, lcb_clx = struct.unpack_from("<II", wd, 0x1A2)
        if lcb_clx == 0 or fc_clx + lcb_clx > len(table):
            raise ParseError("CLX(조각 표)가 없거나 범위를 벗어남 (손상)")
        clx = table[fc_clx : fc_clx + lcb_clx]
        i = 0
        while i < len(clx) and clx[i] == 0x01:  # Prc 건너뛰기
            i += 3 + struct.unpack_from("<H", clx, i + 1)[0]
        if i >= len(clx) or clx[i] != 0x02:
            raise ParseError("PlcPcd 를 찾지 못함 (손상)")
        plc_len = struct.unpack_from("<I", clx, i + 1)[0]
        plc = clx[i + 5 : i + 5 + plc_len]
        n = (plc_len - 4) // 12
        if n <= 0 or len(plc) < 4 * (n + 1) + 8 * n:
            raise ParseError("PlcPcd 길이가 맞지 않음 (손상)")
        cps = struct.unpack_from(f"<{n + 1}i", plc, 0)
        total = ccp_text + ccp_ftn + ccp_hdd + ccp_atn + ccp_edn
        out: list[str] = []
        size = 0
        for k in range(n):
            lo, hi = cps[k], cps[k + 1]
            if hi <= lo or lo >= total:
                continue
            hi = min(hi, total)
            fc = struct.unpack_from("<I", plc, 4 * (n + 1) + 8 * k + 2)[0]
            if fc & 0x40000000:                  # 8비트(cp1252) 조각
                start = (fc & 0x3FFFFFFF) // 2
                raw = wd[start : start + (hi - lo)]
                out.append(raw.decode("cp1252", errors="replace"))
            else:                                # UTF-16LE 조각
                start = fc
                raw = wd[start : start + 2 * (hi - lo)]
                out.append(raw.decode("utf-16-le", errors="replace"))
            size += hi - lo
            if size > MAX_TEXT_CHARS * 2:
                break
        text = "".join(out)
        main = text[:ccp_text]
        ftn = text[ccp_text : ccp_text + ccp_ftn]                                         # 각주
        edn_start = ccp_text + ccp_ftn + ccp_hdd + ccp_atn                                 # 머리글/바닥글·메모(주석)는 건너뛴다
        edn = text[edn_start : edn_start + ccp_edn]                                       # 미주
        body = _doc_clean(main)
        foot = _doc_clean(ftn + "\n" + edn).strip()
        return _clip(body + (("\n[각주/미주]\n" + foot) if foot else ""))
    except ParseError:
        raise
    except struct.error as exc:
        raise ParseError(f"Word 구조 해석 실패(손상): {exc}") from exc
    finally:
        ole.close()


# ---------------------------------------------------------------------------
# .ppt (PowerPoint 97–2003 바이너리, OLE) — 레코드 트리 + 영속 디렉터리로 슬라이드 표시 순서 복원
# ---------------------------------------------------------------------------
_PPT_SLIDE, _PPT_NOTES, _PPT_SLIDE_ATOM = 0x03EE, 0x03F0, 0x03EF
_PPT_SLWT, _PPT_SLIDE_PERSIST = 0x0FF0, 0x03F3
_PPT_TEXT_CHARS, _PPT_TEXT_BYTES = 0x0FA0, 0x0FA8
_PPT_USER_EDIT, _PPT_PERSIST_DIR = 0x0FF5, 0x1772
MAX_PPT_RECORDS = 2_000_000


def _ppt_records(buf: bytes, start: int, end: int, depth: int = 0):
    """(type, ver, inst, data_start, data_end) 을 깊이 우선으로 낸다. 컨테이너(ver==0xF)는 안쪽까지 들어간다."""
    i = start
    while i + 8 <= end:
        ver_inst, typ, ln = struct.unpack_from("<HHI", buf, i)
        ver, inst = ver_inst & 0xF, ver_inst >> 4
        d0, d1 = i + 8, i + 8 + ln
        if d1 > end:
            raise ParseError("PPT 레코드 길이가 범위를 벗어남 (손상)")
        yield typ, ver, inst, d0, d1
        if ver == 0xF:
            if depth > 24:
                raise ParseError("PPT 레코드 중첩이 너무 깊음 (손상)")
            yield from _ppt_records(buf, d0, d1, depth + 1)
        i = d1


_PPT_META_CHARS = {0x0FD8, 0x0FD9, 0x0FDA, 0x0FDB, 0x0FDC}   # 슬라이드 번호·날짜·머리글·바닥글 필드가 글 속에 놓인 자리(position)


def _ppt_decode(raw: bytes, wide: bool) -> str:
    return raw.decode("utf-16-le", errors="replace") if wide else raw.decode("cp1252", errors="replace")


def _ppt_container_text(buf: bytes, d0: int, d1: int) -> list[str]:
    """컨테이너 안의 텍스트 상자들을 순서대로. 필드 자리표시 글자('*' 등)는 MetaChar 위치 정보로 제거한다."""
    out: list[str] = []
    cur: list = []  # [text, [positions]]

    def flush() -> None:
        if not cur:
            return
        text, positions = cur
        for p in sorted(set(positions), reverse=True):
            if 0 <= p < len(text):
                text = text[:p] + text[p + 1 :]
        text = text.replace("\r", "\n").replace("\x0b", "\n").replace("\x00", "").strip()
        if text:
            out.append(text)
        cur.clear()

    for typ, _ver, _inst, a, b in _ppt_records(buf, d0, d1):
        if typ in (_PPT_TEXT_CHARS, _PPT_TEXT_BYTES):
            flush()
            cur.extend([_ppt_decode(buf[a:b], typ == _PPT_TEXT_CHARS), []])
        elif typ in _PPT_META_CHARS and cur and b - a >= 4:
            cur[1].append(struct.unpack_from("<I", buf, a)[0])
    flush()
    return out


def _ppt_persist_map(buf: bytes, cur_user: bytes) -> dict[int, int]:
    """persistId -> 스트림 오프셋. 최신 편집부터 이전 편집으로 거슬러 올라가며, 이미 있는 id 는 덮어쓰지 않는다."""
    if len(cur_user) < 20:
        raise ParseError("Current User 스트림이 너무 짧음 (손상)")
    token = struct.unpack_from("<I", cur_user, 12)[0]
    if token == 0xF3D1C4DF:
        raise ParseError("암호가 걸린 PowerPoint 문서")
    off = struct.unpack_from("<I", cur_user, 16)[0]
    mapping: dict[int, int] = {}
    seen: set[int] = set()
    while off and off not in seen and off + 8 <= len(buf):
        seen.add(off)
        ver_inst, typ, ln = struct.unpack_from("<HHI", buf, off)
        if typ != _PPT_USER_EDIT or off + 8 + ln > len(buf) or ln < 16:
            raise ParseError("UserEditAtom 위치가 올바르지 않음 (손상)")
        last_edit, pdir = struct.unpack_from("<II", buf, off + 8 + 8)
        if pdir + 8 > len(buf):
            raise ParseError("PersistDirectory 위치가 범위를 벗어남 (손상)")
        _, ptyp, pln = struct.unpack_from("<HHI", buf, pdir)
        if ptyp != _PPT_PERSIST_DIR or pdir + 8 + pln > len(buf):
            raise ParseError("PersistDirectoryAtom 이 올바르지 않음 (손상)")
        j, end = pdir + 8, pdir + 8 + pln
        while j + 4 <= end:
            head = struct.unpack_from("<I", buf, j)[0]
            pid, cnt = head & 0xFFFFF, head >> 20
            j += 4
            for k in range(cnt):
                if j + 4 > end:
                    break
                mapping.setdefault(pid + k, struct.unpack_from("<I", buf, j)[0])
                j += 4
        off = last_edit
    return mapping


def _parse_ppt(data: bytes, path: Path) -> str:
    if data[:2] == b"PK":                       # 확장자만 .ppt 이고 실제로는 pptx
        return _parse_pptx(data, path)
    ole = _ole_open(data, "PPT")
    try:
        if not ole.exists("PowerPoint Document"):
            raise ParseError("PowerPoint Document 스트림이 없음 (PowerPoint 문서가 아니거나 손상)")
        buf = ole.openstream("PowerPoint Document").read()
        cur = ole.openstream("Current User").read() if ole.exists("Current User") else b""
        # 1) 표시 순서: SlideListWithText(instance 0) 의 SlidePersistAtom 순서 -> persistId -> 오프셋
        order: list[int] = []                       # 슬라이드(instance 0) 의 persistId, 표시 순서
        notes_by_slide_id: dict[int, int] = {}      # 노트(instance 2): slideId -> persistId  (SlideAtom.notesIdRef 는 slideId 를 가리킨다)
        list_inst = -1
        n_rec = 0
        for typ, ver, inst, a, b in _ppt_records(buf, 0, len(buf)):
            n_rec += 1
            if n_rec > MAX_PPT_RECORDS:
                raise ParseError("PPT 레코드 수가 상한을 넘음")
            if typ == _PPT_SLWT:
                list_inst = inst
            elif typ == _PPT_SLIDE_PERSIST and b - a >= 16:
                persist_id, slide_id = struct.unpack_from("<I", buf, a)[0], struct.unpack_from("<I", buf, a + 12)[0]
                if list_inst == 0:
                    order.append(persist_id)
                elif list_inst == 2:
                    notes_by_slide_id[slide_id] = persist_id
        slides: list[tuple[int, int]] = []         # (오프셋, notesPersistId)
        ok = False
        if cur:
            try:
                pmap = _ppt_persist_map(buf, cur)
                for pid in order:
                    off = pmap.get(pid)
                    if off is None or off + 8 > len(buf):
                        continue
                    _, typ, ln = struct.unpack_from("<HHI", buf, off)
                    if typ == _PPT_SLIDE and off + 8 + ln <= len(buf):
                        slides.append((off, pid))
                ok = bool(slides)
            except ParseError as exc:
                if "암호" in str(exc):
                    raise
        lines: list[str] = []
        if ok:
            for n, (off, _pid) in enumerate(slides, 1):
                _, _t, ln = struct.unpack_from("<HHI", buf, off)
                texts = _ppt_container_text(buf, off + 8, off + 8 + ln)
                notes_id = next((struct.unpack_from("<I", buf, a + 16)[0] for typ, _v, _i, a, b in _ppt_records(buf, off + 8, off + 8 + ln)
                                 if typ == _PPT_SLIDE_ATOM and b - a >= 20), 0)
                lines.append(f"[슬라이드 {n}]")
                lines.extend(texts)
                noff = pmap.get(notes_by_slide_id.get(notes_id, -1)) if notes_id else None
                if noff is not None and noff + 8 <= len(buf):
                    _, ntyp, nln = struct.unpack_from("<HHI", buf, noff)
                    if ntyp == _PPT_NOTES and noff + 8 + nln <= len(buf):
                        nt = _ppt_container_text(buf, noff + 8, noff + 8 + nln)
                        if nt:
                            lines.append("[발표자 노트]")
                            lines.extend(nt)
        else:
            # 영속 디렉터리를 따라가지 못하면 파일 안의 등장 순서로 읽는다(순서가 다를 수 있음)
            n = 0
            for typ, ver, inst, a, b in _ppt_records(buf, 0, len(buf)):
                if typ == _PPT_SLIDE:
                    n += 1
                    lines.append(f"[슬라이드 {n}]")
                    lines.extend(_ppt_container_text(buf, a, b))
        return _clip("\n".join(lines))
    except ParseError:
        raise
    except struct.error as exc:
        raise ParseError(f"PowerPoint 구조 해석 실패(손상): {exc}") from exc
    finally:
        ole.close()


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------
def _parse_eml(data: bytes, path: Path) -> str:
    """.eml 메일 한 통: 제목·보낸이·받는이·날짜 + 본문(text/plain 우선, 없으면 html 을 글자만) + 첨부 파일 이름(내용은 읽지 않는다)."""
    import email
    import email.policy
    import html as _html
    from html.parser import HTMLParser

    msg = email.message_from_bytes(data, policy=email.policy.default)
    head = []
    for key, label in (("Subject", "제목"), ("From", "보낸 사람"), ("To", "받는 사람"), ("Cc", "참조"), ("Date", "날짜")):
        try:
            v = str(msg.get(key, "") or "").strip()
        except Exception:   # 잘못된 헤더 하나가 메일 전체를 버리게 하지 않는다
            v = ""
        if v:
            head.append(f"{label}: {v}")
    body = ""
    part = msg.get_body(preferencelist=("plain", "html"))
    if part is not None:
        try:
            body = part.get_content()
        except Exception:   # 알 수 없는 charset 등: 원문 바이트를 느슨하게 해석
            body = (part.get_payload(decode=True) or b"").decode("utf-8", errors="replace")
        if part.get_content_type() == "text/html":
            class _Strip(HTMLParser):
                def __init__(self):
                    super().__init__(convert_charrefs=True); self.out = []; self.skip = 0
                def handle_starttag(self, tag, attrs):
                    if tag in ("script", "style"): self.skip += 1
                    elif tag in ("br", "p", "div", "tr", "li"): self.out.append("\n")
                def handle_endtag(self, tag):
                    if tag in ("script", "style") and self.skip: self.skip -= 1
                def handle_data(self, d):
                    if not self.skip: self.out.append(d)
            st = _Strip(); st.feed(body); body = _html.unescape("".join(st.out))
    names = [n for n in (p.get_filename() for p in msg.iter_attachments()) if n]
    parts = ["\n".join(head), body.strip()]
    if names:
        parts.append("[첨부] " + ", ".join(names))
    return "\n\n".join(x for x in parts if x).replace("\r\n", "\n")


_PARSERS = {
    ".md": _parse_text,
    ".txt": _parse_text,
    ".pdf": _parse_pdf,
    ".docx": _parse_docx,
    ".xlsx": _parse_xlsx,
    ".pptx": _parse_pptx,
    ".hwpx": _parse_hwpx,
    ".hwp": _parse_hwp,
    ".odt": _parse_odf,
    ".ods": _parse_odf,
    ".odp": _parse_odf,
    ".doc": _parse_doc,
    ".ppt": _parse_ppt,
    ".eml": _parse_eml,
}
SUPPORTED_EXTENSIONS = frozenset(_PARSERS)
BINARY_EXTENSIONS = frozenset(SUPPORTED_EXTENSIONS - {".md", ".txt"})
OFFICE_EXTENSIONS = frozenset({".docx", ".xlsx", ".pptx", ".hwpx", ".hwp", ".odt", ".ods", ".odp", ".doc", ".ppt", ".eml"})  # loaders 가 이 파서로 읽는 형식 (pdf/md/txt 는 기존 경로 유지)


def can_parse(path: Path) -> bool:
    return path.suffix.lower() in _PARSERS


def extract_text(path: Path, data: bytes | None = None) -> str:
    """텍스트를 추출한다. 지원하지 않는 확장자/읽을 수 없는 파일은 ParseError.
    data 를 주면 파일을 다시 읽지 않는다 (색인 때 해시를 낸 바로 그 바이트를 파싱하기 위해)."""
    fn = _PARSERS.get(path.suffix.lower())
    if fn is None:
        raise ParseError(f"지원하지 않는 형식: {path.suffix or '(확장자 없음)'}")
    if data is None:
        data = path.read_bytes()
    try:
        text = fn(data, path)
        if path.suffix.lower() not in (".md", ".txt"):   # 일반 텍스트는 쓴 그대로 둔다
            for k, v in _LIGATURES.items():
                text = text.replace(k, v)
        return text
    except ParseError:
        raise
    except Exception as exc:  # noqa: BLE001 - 파서 내부의 예기치 못한 오류도 사유와 함께 ParseError 로 통일
        raise ParseError(f"{type(exc).__name__}: {exc}") from exc


def extract_text_or_none(path: Path) -> str | None:
    """live_semantic / 가드용: 실패는 None (호출자가 '읽지 못한 파일'로 보고한다)."""
    try:
        return extract_text(path) or None
    except (ParseError, OSError):
        return None

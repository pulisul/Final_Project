"""실문서가 제대로 읽히는지 사람이 확인할 수 있게 돕는 점검기.

왜 필요한가: 파서 테스트는 우리가 만든 합성 파일로 했다. 포맷을 잘못 이해했다면 파서와 테스트가 함께 틀려도 통과한다.
이 모듈은 *실제 파일*(한컴 .hwp/.hwpx, 워드, PDF 등)에 대해
 (1) 추출 텍스트가 깨졌는지(휴리스틱), (2) 사람이 적어 둔 '반드시 들어 있어야 할 문장' 이 들어 있는지를 본다.
(2) 가 이 점검기의 핵심 증거다: 원본 문서를 열어 눈으로 고른 문장이 추출문에 있으면, 합성 파일이 아닌 실문서에서 맞다는 뜻이다.

개인정보: 보고서에는 기본적으로 본문을 싣지 않는다(해시·길이·지표만). 본문을 보려면 ``show_chars`` 또는 덤프 폴더를 명시해야 한다.
"""
from __future__ import annotations

import hashlib
import io
import re
import struct
import time
import unicodedata
import zipfile
from pathlib import Path
from typing import Sequence

from . import doc_parsers
from .loaders import SafeDocumentLoader

ALL_EXTS = (".md", ".txt", ".pdf", *sorted(doc_parsers.OFFICE_EXTENSIONS))
MAX_BYTES = 200 * 1024 * 1024

_HANGUL = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")
_CJK = re.compile(r"[一-鿿]")
_PUA = re.compile(r"[-]")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZW = re.compile(r"[​-‏⁠﻿]")


def normalize_for_match(s: str) -> str:
    """비교용 정규화: NFC, 모든 공백(줄바꿈·탭·nbsp 포함) 제거. 한글 문서는 줄바꿈 위치가 달라도 같은 문장이어야 한다."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", s))


def text_quality(text: str) -> dict:
    """텍스트가 '글처럼' 생겼는지 보는 지표와 의심 플래그. 판정은 휴리스틱이므로 플래그는 *의심*이지 확정이 아니다."""
    nonspace = re.sub(r"\s+", "", text)
    n = len(nonspace)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    m = {"chars": len(text), "nonspace_chars": n, "lines": len(lines)}
    if n == 0:
        m.update(hangul_ratio=0.0, cjk_ratio=0.0, pua_count=0, replacement_count=0, control_count=0, zero_width_count=0,
                 avg_line_len=0.0, duplicate_line_ratio=0.0)
        m["flags"] = ["EMPTY"]
        return m
    hangul = len(_HANGUL.findall(nonspace))
    cjk = len(_CJK.findall(nonspace))
    pua = len(_PUA.findall(text))
    repl = text.count("�")
    ctrl = len(_CTRL.findall(text))
    zw = len(_ZW.findall(text))
    uniq = len(set(lines))
    m.update(hangul_ratio=round(hangul / n, 4), cjk_ratio=round(cjk / n, 4), pua_count=pua, replacement_count=repl,
             control_count=ctrl, zero_width_count=zw, avg_line_len=round(sum(len(x) for x in lines) / len(lines), 1),
             duplicate_line_ratio=round(1 - uniq / len(lines), 4) if lines else 0.0)
    flags: list[str] = []
    if repl / n > 0.001:
        flags.append("REPLACEMENT_CHARS")          # 디코딩 실패 흔적(�)
    if pua / n > 0.01:
        flags.append("PRIVATE_USE_HIGH")           # 한컴 옛한글/글머리 PUA 가 많음: 일부 글자가 다른 코드로 읽혔을 수 있음
    if cjk / n > 0.30 and hangul / n < 0.05:
        flags.append("CJK_WITHOUT_HANGUL")         # 한글 문서인데 한자만 많음 → 인코딩 오독(모지바케) 전형
    if ctrl / n > 0.001:
        flags.append("CONTROL_CHARS")
    if len(lines) >= 20 and m["avg_line_len"] < 8:
        flags.append("ONE_WORD_PER_LINE")          # PDF 단어당 줄바꿈 등: 문장이 깨져 검색이 어렵다
    if len(lines) >= 20 and m["duplicate_line_ratio"] > 0.5:
        flags.append("MANY_DUPLICATE_LINES")       # 머리글/바닥글 반복 또는 표 반복
    m["flags"] = flags
    return m


def read_expected(path: Path, expect_dir: Path | None = None) -> list[str]:
    """``<파일>.expect.txt`` (또는 expect_dir 의 같은 이름) 에서 문장 목록. 한 줄에 하나, 빈 줄과 ``#`` 시작 줄은 무시."""
    cands = [path.with_name(path.name + ".expect.txt")]
    if expect_dir is not None:
        cands.insert(0, expect_dir / (path.name + ".expect.txt"))
    for c in cands:
        if c.is_file():
            raw = c.read_text(encoding="utf-8-sig")
            return [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    return []


def check_expected(text: str, phrases: Sequence[str]) -> dict:
    """사람이 원본에서 고른 문장이 추출문에 있는지. 공백·줄바꿈 차이는 무시한다."""
    hay = normalize_for_match(text)
    found, missing = [], []
    for p in phrases:
        needle = normalize_for_match(p)
        (found if needle and needle in hay else missing).append(p)
    return {"total": len(phrases), "found": len(found), "missing": missing,
            "recall": round(len(found) / len(phrases), 4) if phrases else None}


# ---- 형식별 진단 ---------------------------------------------------------------------------------------------------------
def hwp_diagnostics(data: bytes) -> dict:
    """HWP 5.x 파일의 구조 요약(본문 없이). olefile 이 없으면 사유만 돌려준다."""
    try:
        import olefile
    except ImportError:
        return {"format": "hwp5", "error": "olefile 미설치 (pip install olefile)"}
    try:
        ole = olefile.OleFileIO(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        return {"format": "hwp5", "error": f"OLE 열기 실패: {type(exc).__name__}: {exc}"}
    try:
        info: dict = {"format": "hwp5", "streams": sorted("/".join(s) for s in ole.listdir())[:40]}
        if ole.exists("FileHeader"):
            h = ole.openstream("FileHeader").read()
            info["signature_ok"] = h.startswith(b"HWP Document File")
            if len(h) >= 40:
                ver = struct.unpack_from("<I", h, 32)[0]
                flags = struct.unpack_from("<I", h, 36)[0]
                info["version"] = f"{(ver >> 24) & 0xFF}.{(ver >> 16) & 0xFF}.{(ver >> 8) & 0xFF}.{ver & 0xFF}"
                info["compressed"] = bool(flags & 1)
                info["encrypted"] = bool(flags & 2)
                info["distribution"] = bool(flags & 4)   # 배포용(열람 제한) 문서
                info["has_script"] = bool(flags & 8)
                info["drm"] = bool(flags & 16)
        else:
            info["signature_ok"] = False
        sections = [s for s in ole.listdir() if len(s) == 2 and s[0] == "BodyText"]
        info["sections"] = len(sections)
        info["has_viewtext"] = any(s and s[0] == "ViewText" for s in ole.listdir())
        return info
    finally:
        ole.close()


def hwpx_diagnostics(data: bytes) -> dict:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        return {"format": "hwpx", "error": f"ZIP 열기 실패: {exc}"}
    names = zf.namelist()
    sections = sorted(n for n in names if re.match(r"Contents/section\d+\.xml$", n))
    return {"format": "hwpx", "entries": len(names), "sections": len(sections),
            "has_manifest": "META-INF/manifest.xml" in names, "has_header": "Contents/header.xml" in names,
            "mimetype": zf.read("mimetype").decode("ascii", "replace").strip() if "mimetype" in names else None}


def format_diagnostics(path: Path, data: bytes) -> dict:
    ext = path.suffix.lower()
    if ext == ".hwp":
        return hwp_diagnostics(data)
    if ext == ".hwpx":
        return hwpx_diagnostics(data)
    if ext == ".pdf":
        head = data[:5]
        return {"format": "pdf", "signature_ok": head == b"%PDF-", "encrypted_marker": b"/Encrypt" in data[-4096:] or b"/Encrypt" in data[:4096]}
    return {"format": ext.lstrip(".")}


def snippets(text: str, n: int, width: int = 60) -> list[str]:
    """눈으로 대조할 토막 n 개(앞·중간·뒤에서 고르게). 본문이 출력되므로 호출자가 명시적으로 요청했을 때만 쓴다."""
    flat = re.sub(r"\s+", " ", text).strip()
    if n <= 0 or not flat:
        return []
    if len(flat) <= width:
        return [flat]
    starts = [round(i * (len(flat) - width) / max(n - 1, 1)) for i in range(n)] if n > 1 else [0]
    return [flat[s:s + width] for s in starts]


def inspect_file(path: Path, *, expect: Sequence[str] | None = None, expect_dir: Path | None = None,
                 max_bytes: int = MAX_BYTES, show_chars: int = 0, sample: int = 0, dump_dir: Path | None = None) -> dict:
    """한 파일을 우리 로더로 읽어(=실제 색인 경로) 결과를 점검한다. 예외는 FAIL 로 기록하고 던지지 않는다."""
    path = Path(path)
    rec: dict = {"name": path.name, "ext": path.suffix.lower(), "bytes": path.stat().st_size if path.exists() else None}
    try:
        data = path.read_bytes()
    except OSError as exc:
        rec.update(status="FAIL", error=f"읽기 실패: {exc}")
        return rec
    rec["sha256_16"] = hashlib.sha256(data).hexdigest()[:16]
    rec["diagnostics"] = format_diagnostics(path, data)
    t0 = time.perf_counter()
    try:
        loader = SafeDocumentLoader(path.parent, ALL_EXTS, max_bytes)
        doc = loader.load(path)
        text = doc.text
        rec["parse_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    except Exception as exc:  # noqa: BLE001 - 어떤 실패도 사유와 함께 보고
        rec.update(status="FAIL", error=f"{type(exc).__name__}: {exc}", parse_ms=round((time.perf_counter() - t0) * 1000, 1))
        return rec
    q = text_quality(text)
    rec["quality"] = q
    if rec["bytes"]:
        rec["chars_per_kb"] = round(len(text) / (rec["bytes"] / 1024), 2)
    phrases = list(expect) if expect is not None else read_expected(path, expect_dir)
    exp = check_expected(text, phrases) if phrases else None
    rec["expected"] = exp
    problems = list(q["flags"])
    if exp and exp["missing"]:
        problems.append(f"EXPECTED_MISSING({len(exp['missing'])}/{exp['total']})")
    rec["problems"] = problems
    if q["flags"] == ["EMPTY"]:
        rec["status"] = "FAIL"
    elif problems:
        rec["status"] = "SUSPECT"
    else:
        rec["status"] = "OK" if exp else "OK_UNVERIFIED"   # 기대 문장이 없으면 '깨지진 않은 듯' 일 뿐 정답 증거는 아니다
    if show_chars:
        rec["head"] = re.sub(r"\s+", " ", text[:show_chars])
    if sample:
        rec["samples"] = snippets(text, sample)
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        (dump_dir / (path.name + ".extracted.txt")).write_text(text, encoding="utf-8")
    return rec


def collect_files(paths: Sequence[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out += sorted(f for f in p.rglob("*") if f.is_file() and not f.is_symlink() and f.suffix.lower() in ALL_EXTS
                          and not f.name.endswith(".expect.txt"))
        elif p.is_file():
            out.append(p)
    return out


def summarize(records: Sequence[dict]) -> dict:
    by_status: dict[str, int] = {}
    by_ext: dict[str, dict[str, int]] = {}
    for r in records:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        by_ext.setdefault(r["ext"], {}).setdefault(r["status"], 0)
        by_ext[r["ext"]][r["status"]] += 1
    verified = [r for r in records if r.get("expected") and r["expected"]["total"]]
    return {"files": len(records), "by_status": by_status, "by_ext": by_ext, "verified_files": len(verified),
            "verified_ok": sum(1 for r in verified if r["status"] == "OK")}

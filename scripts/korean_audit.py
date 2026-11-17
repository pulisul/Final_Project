"""한글 처리 점검: 추출 텍스트(evidence/real_docs_dump/*.extracted.txt)와 문서 폴더 파일명을 검사한다.
사용(프로젝트 루트에서): .venv-real/bin/python korean_audit.py [덤프폴더] [문서폴더]
판정은 휴리스틱이다. 플래그는 '의심'이지 확정이 아니므로, 플래그가 뜬 파일은 덤프를 열어 눈으로 확인한다."""
import re, sys, unicodedata
from collections import Counter
from pathlib import Path

dump = Path(sys.argv[1] if len(sys.argv) > 1 else "evidence/real_docs_dump")
docs = Path(sys.argv[2] if len(sys.argv) > 2 else "data/real_docs")
JAMO = re.compile("[ᄀ-ᇿ㄰-㆏ꥠ-꥿ힰ-퟿]")
SYL = re.compile("[가-힣]")
SPACED = re.compile(r"(?:(?<![가-힣])[가-힣] ){4,}[가-힣](?![가-힣])")   # '한 글 처 리' 처럼 글자마다 띄어진 구간
rows, flags = [], []
for f in sorted(dump.glob("*.extracted.txt")):
    name = f.name.replace(".extracted.txt", "")
    if docs.exists() and not (docs / name).exists():
        continue                                        # 이미 치운 문서의 낡은 덤프는 제외
    t = f.read_text(encoding="utf-8", errors="replace")
    n_syl, n_jamo = len(SYL.findall(t)), len(JAMO.findall(t))
    fffd = t.count("�"); pua = sum(1 for c in t if "" <= c <= ""); zw = sum(t.count(c) for c in "​‌‍﻿")
    spaced = len(SPACED.findall(t))
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    cnt = Counter(l for l in lines if len(l) >= 5)
    rep = sum(v for v in cnt.values() if v >= 5) / max(1, len(lines))
    short = sum(1 for l in lines if len(l) <= 3) / max(1, len(lines))
    nfc_diff = t != unicodedata.normalize("NFC", t)
    fl = []
    if n_jamo and n_jamo / max(1, n_syl + n_jamo) > 0.005: fl.append("NFD/자모분리")
    if nfc_diff and not fl: fl.append("NFC아님(조합형 혼재)")
    if fffd: fl.append(f"깨진글자{fffd}")
    if pua: fl.append(f"사용자정의영역{pua}")
    if spaced: fl.append(f"글자띄어쓰기{spaced}곳")
    if rep > 0.15: fl.append(f"반복줄{rep:.0%}")
    if short > 0.25: fl.append(f"짧은줄{short:.0%}")
    rows.append((name, len(t), n_syl, n_jamo, fffd, spaced, f"{rep:.0%}", f"{short:.0%}", ",".join(fl) or "-"))
print(f"{'파일':<42}{'글자':>7}{'한글':>7}{'자모':>6}{'깨짐':>5}{'띄움':>5}{'반복':>6}{'짧은':>6}  플래그")
for r in rows:
    print(f"{r[0][:40]:<42}{r[1]:>7}{r[2]:>7}{r[3]:>6}{r[4]:>5}{r[5]:>5}{r[6]:>6}{r[7]:>6}  {r[8]}")
nfd_names = [p.name for p in docs.rglob("*") if p.is_file() and p.name != unicodedata.normalize("NFC", p.name)] if docs.exists() else []
print("\n파일명이 NFD(맥 분리형)인 문서:", nfd_names or "없음")
flagged = [r for r in rows if r[8] != "-"]
print(f"\n점검 {len(rows)}개 중 의심 {len(flagged)}개." + ("" if flagged else " 한글 추출 이상 징후 없음(휴리스틱 기준)."))

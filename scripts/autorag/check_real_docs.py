"""실문서(특히 한컴 .hwp/.hwpx, 워드, PDF)가 제대로 읽히는지 점검한다. 모델·네트워크 불필요.

사용 (문서가 있는 컴퓨터에서):
  PYTHONPATH=src python scripts/autorag/check_real_docs.py <파일 또는 폴더>... --out evidence/real_docs_check
옵션
  --expect-dir DIR   '<파일명>.expect.txt' 를 찾을 폴더(기본: 문서 옆). 한 줄에 하나, 원본에서 눈으로 고른 *실제 문장*.
  --dump-dir DIR     추출 본문을 파일로 저장(원본과 나란히 열어 대조). 보고서에는 본문을 싣지 않는다.
  --show-chars N     화면/보고서에 본문 앞 N 자를 싣는다(기본 0). --sample N 은 앞·중간·뒤 토막 N 개.

판정
  OK            깨짐 징후 없음 + 기대 문장이 모두 들어 있음  ← 이것만 '실문서에서 맞다' 의 증거
  OK_UNVERIFIED 깨짐 징후는 없으나 기대 문장을 주지 않음      ← 증거 아님. expect.txt 를 만들어 다시 돌릴 것
  SUSPECT       깨짐 의심 플래그 또는 기대 문장 누락
  FAIL          읽지 못함 / 빈 텍스트
종료 코드: 0 = 모두 OK 계열, 1 = SUSPECT/FAIL 있음, 2 = 점검할 파일 없음

보고서(report.json/report.md)에는 파일명·해시·지표만 들어간다. 보고서를 공유해도 문서 내용은 새지 않는다(show-chars/sample 을 쓰지 않았다면).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from intellectagent.doc_check import collect_files, inspect_file, summarize


def render_md(records: list[dict], summ: dict) -> str:
    L = ["# 실문서 점검 보고서", "",
         f"- 파일 {summ['files']}개 / 상태 {json.dumps(summ['by_status'], ensure_ascii=False)} / 기대 문장으로 검증한 파일 {summ['verified_files']}개 (OK {summ['verified_ok']})",
         "- **OK 만 실문서 증거.** OK_UNVERIFIED 는 '깨지진 않은 듯' 일 뿐이다.", "",
         "| 파일 | 형식 | 상태 | 글자수 | 한글비율 | 기대문장 | 문제 |", "|---|---|---|---:|---:|---|---|"]
    for r in records:
        q = r.get("quality") or {}
        e = r.get("expected")
        exp = f"{e['found']}/{e['total']}" if e else "-"
        L.append(f"| {r['name']} | {r['ext']} | {r['status']} | {q.get('chars', '-')} | {q.get('hangul_ratio', '-')} | {exp} | "
                 f"{', '.join(r.get('problems') or [r.get('error', '')]) or '-'} |")
    for r in records:
        d = r.get("diagnostics") or {}
        if d.get("format") == "hwp5":
            L.append(f"\n- `{r['name']}` HWP 구조: 버전 {d.get('version')}, 압축 {d.get('compressed')}, 암호 {d.get('encrypted')}, 배포용 {d.get('distribution')}, 섹션 {d.get('sections')}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+"); ap.add_argument("--out", default=None)
    ap.add_argument("--expect-dir", default=None); ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--show-chars", type=int, default=0); ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--max-file-mb", type=int, default=200)
    a = ap.parse_args(argv)
    files = collect_files([Path(p) for p in a.paths])
    if not files:
        print("점검할 파일이 없습니다 (지원 확장자: .md .txt .pdf .docx .xlsx .pptx .hwpx .hwp .doc .ppt .odt .ods .odp .eml).", file=sys.stderr)
        return 2
    exp_dir = Path(a.expect_dir) if a.expect_dir else None
    dump = Path(a.dump_dir) if a.dump_dir else None
    recs = [inspect_file(f, expect_dir=exp_dir, max_bytes=a.max_file_mb * 1024 * 1024, show_chars=a.show_chars, sample=a.sample, dump_dir=dump) for f in files]
    summ = summarize(recs)
    md = render_md(recs, summ)
    print(md)
    for r in recs:
        if r.get("head"):
            print(f"[{r['name']}] 앞부분: {r['head']}")
        for s in r.get("samples", []):
            print(f"[{r['name']}] 토막: {s}")
    if a.out:
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps({"summary": summ, "files": recs}, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "report.md").write_text(md, encoding="utf-8")
        print(f"저장: {out}/report.md, report.json")
    return 0 if all(r["status"] in ("OK", "OK_UNVERIFIED") for r in recs) else 1


if __name__ == "__main__":
    sys.exit(main())

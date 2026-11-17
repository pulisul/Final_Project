"""여러 폴더를 *색인하지 않고* 훑어서, 무엇이 색인될지 미리 보여 준다 (읽기 전용).

  # 후보 폴더 제안 (Documents/Downloads/Desktop 등의 지원 파일 개수만 센다. 아무것도 색인하지 않음)
  PYTHONPATH=src python scripts/autorag/scan_corpus.py --suggest
  # 승인한 폴더 점검: 제외된 것과 이유, 중복, 확장자별 개수, 지난번 대비 추가/수정/삭제
  PYTHONPATH=src python scripts/autorag/scan_corpus.py --roots ~/Documents,~/Downloads --manifest evidence/corpus_manifest.json
홈 폴더 전체는 --allow-home 을 줘야 하고, 파일시스템 루트와 시스템 폴더는 어떤 옵션으로도 거부된다.
실제 색인은 config 의 feature_flags.search_roots 에 같은 폴더를 넣고 기존 색인 명령/Streamlit 으로 한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from intellectagent.corpus_scan import (RootRejected, build_manifest, dedupe_exact, diff_manifest, load_manifest, save_manifest,
                                        scan_roots, suggest_candidate_roots)
from intellectagent.doc_parsers import OFFICE_EXTENSIONS

EXTS = (".md", ".txt", ".pdf", *sorted(OFFICE_EXTENSIONS))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suggest", action="store_true"); ap.add_argument("--roots", default="")
    ap.add_argument("--allow-home", action="store_true"); ap.add_argument("--max-file-mb", type=int, default=60)
    ap.add_argument("--no-dedupe", action="store_true"); ap.add_argument("--manifest", default=None)
    ap.add_argument("--no-save-manifest", action="store_true"); ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.suggest:
        cands = suggest_candidate_roots(EXTS)
        print(json.dumps(cands, ensure_ascii=False, indent=2) if a.json else
              "\n".join(f"{c['path']}  지원 파일 {c['supported_files']}{'+' if c['truncated'] else ''}개" for c in cands) or "후보 폴더가 없습니다.")
        print("→ 이 중 색인해도 되는 폴더만 --roots 로 직접 지정하세요.")
        return 0
    if not a.roots:
        print("--roots 또는 --suggest 가 필요합니다. (PC 전체를 자동으로 훑지 않습니다)", file=sys.stderr)
        return 2
    try:
        scan = scan_roots([Path(r.strip()) for r in a.roots.split(",") if r.strip()], EXTS, a.max_file_mb * 1024 * 1024, allow_home=a.allow_home)
    except RootRejected as exc:
        print(f"거부: {exc}", file=sys.stderr)
        return 2
    kept, dups = (scan.entries, []) if a.no_dedupe else dedupe_exact(scan.entries)
    by_ext = Counter(e.path.suffix.lower() for e in kept)
    saved = sum(d.size for d, _ in dups)
    report = {"roots": [str(r) for r in scan.roots], "files_to_index": len(kept), "bytes": sum(e.size for e in kept),
              "by_extension": dict(by_ext.most_common()), "duplicates_dropped": len(dups), "duplicate_bytes_saved": saved,
              "skipped": dict(scan.skipped), "skipped_examples": scan.examples, "truncated": scan.truncated,
              "duplicate_examples": [{"dropped": str(d.path), "kept": str(k.path)} for d, k in dups[:5]]}
    if a.manifest:
        new = build_manifest(kept)
        old = load_manifest(Path(a.manifest))
        report["changes_since_last"] = ({k: len(v) for k, v in diff_manifest(old, new).items()} if old else "첫 실행")
        if not a.no_save_manifest:
            save_manifest(Path(a.manifest), new)
    if a.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"루트: {', '.join(report['roots'])}\n색인 대상 {report['files_to_index']}개 ({report['bytes'] / 1e6:.1f} MB)  중복 제외 {len(dups)}개 ({saved / 1e6:.1f} MB)")
        print("확장자별:", ", ".join(f"{k} {v}" for k, v in by_ext.most_common()) or "-")
        for reason, n in scan.skipped.most_common():
            ex = f"  예: {scan.examples[reason][0]}" if scan.examples.get(reason) else ""
            print(f"  건너뜀 {n:>6}  {reason}{ex}")
        if "changes_since_last" in report:
            print("지난번 대비:", report["changes_since_last"])
        if scan.truncated:
            print("⚠ 파일이 너무 많아 일부만 스캔했습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

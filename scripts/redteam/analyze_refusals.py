#!/usr/bin/env python3
"""Mac 에서 실행: 종단 평가에서 '근거 검증 실패'로 거절된 답변의 원문을 보고, 왜 막혔는지 분류한다.
사용(프로젝트 루트): .venv-real/bin/python analyze_refusals.py evidence/eval_final/e2e_samples.jsonl
읽기 전용. 모델·네트워크 불필요."""
import json, re, sys
from collections import Counter
from pathlib import Path

rows = [json.loads(l) for l in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if l.strip()]
CITE = re.compile(r"\[(\d{1,3}(?:\s*[,，]\s*\d{1,3})*)\]|\[[0-9a-f]{20}\]")


def kind(ln: str) -> str:
    t = ln.strip()
    if re.match(r"^#{1,6}\s", t): return "제목줄(#)"
    if re.match(r"^\*\*.*\*\*:?$", t) or (len(t) <= 20 and t.endswith(("*", "**"))): return "굵은 머리말"
    if len(t) <= 20 and t.endswith((":", "：")): return "짧은 머리말(:)"
    if re.match(r"^(\d+[.)]|[-*•])\s*$", t): return "빈 글머리"
    if not CITE.search(t) and len(t) <= 12: return "짧은 줄(12자 이하)"
    return "인용 없는 주장 줄"


bad = [r for r in rows if r.get("refusal_reason") == "grounding_failed"]
print(f"근거 검증 실패 {len(bad)}건 / 전체 {len(rows)}건")
tot = Counter(); would_pass = 0; detail = []
for r in bad:
    raw = r.get("raw_answer", "") or ""
    lines = [l for l in raw.splitlines() if l.strip()]
    uncited = [l for l in lines if not CITE.search(l)]
    kinds = Counter(kind(l) for l in uncited)
    only_format = bool(uncited) and all(k != "인용 없는 주장 줄" for k in kinds) and any(CITE.search(l) for l in lines)
    g = r.get("grounding") or {}
    reason = ("인용이 하나도 없음" if not any(CITE.search(l) for l in lines) else
              "잘못된 번호/ID 인용" if g.get("bad_citations") else
              "서식 줄 때문에만 막힘(내용 줄은 모두 인용)" if only_format else "내용 줄에 인용 누락")
    tot[reason] += 1; would_pass += only_format
    detail.append((r.get("case_id"), "답 없어야 함" if r.get("expected_fallback") else "답 있어야 함", reason, dict(kinds)))
for k, v in tot.most_common(): print(f"  {v:3d}  {k}")
print(f"\n서식 줄만 문제인 건수: {would_pass}  (이 규칙을 완화하면 통과할 후보 — 단, 답 없어야 하는 질문이면 오히려 통과하면 안 됨)")
print("\n사례:")
for d in detail: print("  ", d)

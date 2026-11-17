"""질문 초안 검수 도구: 답 있는 draft 질문을 하나씩 보여 주고 승인(y)/제외(n)/건너뜀(s)/종료(q)를 받는다.
사용: .venv-real/bin/python review_qa.py [evidence/qa/qa_draft.jsonl]
- 승인(y) 전에 '근거'에 '정답'이 실제로 들어 있는지, 질문이 근거 없이도 뜻이 통하는지 눈으로 확인한다.
- 입력할 때마다 파일에 바로 저장하므로 중간에 종료해도 이어서 할 수 있다(이미 승인/제외한 줄은 다시 안 나온다)."""
import json, os, sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "evidence/qa/qa_draft.jsonl")
rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
todo = [i for i, r in enumerate(rows) if r.get("status") == "draft" and not r.get("should_fallback")]


def save():
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, path)


def count():
    ok = [r for r in rows if r.get("status") == "approved"]
    return sum(1 for r in ok if not r.get("should_fallback")), sum(1 for r in ok if r.get("should_fallback"))


print(f"검수할 질문 {len(todo)}개 (y 승인 / n 제외 / s 건너뜀 / q 종료)")
for k, i in enumerate(todo, 1):
    r = rows[i]
    print("\n" + "=" * 70)
    print(f"[{k}/{len(todo)}] 유형 {r.get('qtype') or '-'}")
    print("질문:", r["question"])
    print("정답:", r.get("answer", ""))
    print("근거:", (r.get("evidence", "") or "")[:500].replace("\n", " "))
    while True:
        a = input("승인? [y/n/s/q] ").strip().lower()
        if a in ("y", "n", "s", "q"):
            break
    if a == "q":
        break
    if a == "y":
        r["status"] = "approved"
    elif a == "n":
        r["status"], r["reason"] = "rejected", "manual_review"
    save()
save()
print("\n승인 현황: answerable %d / unanswerable %d" % count())

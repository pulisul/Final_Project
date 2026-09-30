from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
out = ROOT / "evidence" / "week8" / "user_test.csv"
out.parent.mkdir(parents=True, exist_ok=True)
fields = ["tester_id", "case_no", "question", "answer_useful_1_5", "source_useful_1_5", "fallback_correct", "reason", "consent_confirmed"]
with out.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for tester in ("tester01", "tester02", "tester03"):
        for n in range(1, 6):
            w.writerow({"tester_id": tester, "case_no": n, "consent_confirmed": ""})
print(out)

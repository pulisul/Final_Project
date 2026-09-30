from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

FIELDS = ["accuracy", "groundedness", "question_satisfaction", "source_utility", "readability"]


def write_human_eval_template(raw_log_path: Path, out_csv: Path) -> None:
    rows = [json.loads(x) for x in raw_log_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "evaluator_id", *FIELDS, "comment"])
        for row in rows:
            writer.writerow([row["case_id"], "", "", "", "", "", "", ""])


def summarize_human_eval(path: Path) -> dict[str, Any]:
    vals = {k: [] for k in FIELDS}
    completed = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            scores = []
            for field in FIELDS:
                raw = (row.get(field) or "").strip()
                if not raw:
                    scores = []
                    break
                score = float(raw)
                if not 1 <= score <= 5:
                    raise ValueError(f"{field} must be 1..5")
                scores.append(score)
            if scores:
                completed += 1
                for field, score in zip(FIELDS, scores, strict=True):
                    vals[field].append(score)
    return {
        "completed_rows": completed,
        "means": {k: (statistics.mean(v) if v else None) for k, v in vals.items()},
    }

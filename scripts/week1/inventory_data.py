from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path

ALLOWED = {".md", ".txt", ".pdf"}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=Path("evidence/week1/data_inventory.json"))
    args = ap.parse_args()
    root = args.root.resolve()
    rows, counts = [], Counter()
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        counts[suffix or "<none>"] += 1
        if suffix not in ALLOWED:
            continue
        stat = p.stat()
        rows.append({
            "relative_path": str(p.relative_to(root)),
            "suffix": suffix,
            "bytes": stat.st_size,
            "sha256": sha256(p),
            "contains_personal_data": "UNKNOWN",
            "notes": "",
        })
    payload = {
        "root": str(root),
        "allowed_extensions": sorted(ALLOWED),
        "file_type_counts": dict(counts),
        "supported_file_count": len(rows),
        "supported_total_bytes": sum(r["bytes"] for r in rows),
        "files": rows,
    }
    out = args.out
    if not out.is_absolute():
        out = Path.cwd() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "files"}, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

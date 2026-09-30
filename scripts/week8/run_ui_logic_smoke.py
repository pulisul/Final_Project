from __future__ import annotations
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))
from intellectagent.ui_support import FeedbackStore, build_cache_fingerprint, load_policy_info, scrub_pii

policy_path = ROOT / "artifacts" / "week7" / "active_policy.json"
config_path = ROOT / "config" / "core.json"
policy = load_policy_info(policy_path)
with tempfile.TemporaryDirectory() as td:
    store = FeedbackStore(Path(td) / "feedback.sqlite3")
    first = store.submit(session_id="s1", question="q", response="a", thumb=1, policy=policy, reason="mail me at abc@example.com")
    second = store.submit(session_id="s1", question="q", response="a", thumb=-1, policy=policy)
    rows = store.rows()
    result = {
        "policy": policy.__dict__,
        "cache_fingerprint": build_cache_fingerprint(config_path=config_path, policy_path=policy_path),
        "feedback_first_inserted": first,
        "feedback_duplicate_inserted": second,
        "feedback_count": store.count(),
        "reason_scrubbed": "abc@example.com" not in rows[0]["reason"],
        "pii_smoke": scrub_pii("abc@example.com / 010-1234-5678"),
    }
    result["pass"] = bool(first and not second and store.count() == 1 and result["reason_scrubbed"])

out = ROOT / "evidence" / "week8" / "ui_logic_smoke.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
raise SystemExit(0 if result["pass"] else 1)

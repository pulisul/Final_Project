from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EMAIL_RE = re.compile(r"(?<![\w.+-])([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})(?![\w.-])")
PHONE_RE = re.compile(r"(?<!\d)(01[016789])[- ]?(\d{3,4})[- ]?(\d{4})(?!\d)")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_json_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def scrub_pii(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL_REDACTED]", text)
    text = PHONE_RE.sub("[PHONE_REDACTED]", text)
    return text


@dataclass(frozen=True)
class PolicyInfo:
    name: str
    version: str
    rollback_applied: bool
    reason: str


def load_policy_info(path: Path) -> PolicyInfo:
    if not path.exists():
        return PolicyInfo("baseline", "missing", False, "policy_file_missing")
    data = json.loads(path.read_text(encoding="utf-8"))
    name = str(data.get("active_policy", "baseline"))
    schema = str(data.get("schema_version", 1))
    digest = sha256_file(path)[:12]
    return PolicyInfo(name=name, version=f"schema{schema}-{digest}", rollback_applied=bool(data.get("rollback_applied", False)), reason=str(data.get("reason", "")))


def build_cache_fingerprint(*, config_path: Path, policy_path: Path, index_marker: Path | None = None) -> str:
    payload = {
        "config_sha256": sha256_file(config_path) if config_path.exists() else "missing",
        "policy_sha256": sha256_file(policy_path) if policy_path.exists() else "missing",
        "index_sha256": sha256_file(index_marker) if index_marker and index_marker.exists() else "missing",
    }
    return stable_json_hash(payload)[:20]


@dataclass(frozen=True)
class HealthStatus:
    component: str
    ok: bool
    detail: str


def inspect_runtime_health(project_root: Path, *, require_index: bool = False) -> list[HealthStatus]:
    checks: list[HealthStatus] = []
    cfg = project_root / "config" / "core.json"
    policy = project_root / "artifacts" / "week7" / "active_policy.json"
    checks.append(HealthStatus("config", cfg.exists(), str(cfg)))
    checks.append(HealthStatus("active_policy", policy.exists(), str(policy)))
    if require_index:
        idx = project_root / "index"
        checks.append(HealthStatus("index", idx.exists(), str(idx)))
    return checks


class FeedbackStore:
    """SQLite feedback store with idempotent submit semantics.

    A feedback_key must be stable per rendered answer. UNIQUE(feedback_key) prevents
    repeated clicks / reruns from inserting duplicate training signals.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self):
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _init_db(self) -> None:
        with self._conn() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feedback_key TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    question_hash TEXT NOT NULL,
                    response_hash TEXT NOT NULL,
                    thumb INTEGER NOT NULL CHECK(thumb IN (-1, 1)),
                    source_helpful INTEGER,
                    reason TEXT NOT NULL DEFAULT '',
                    policy_name TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    @staticmethod
    def make_key(session_id: str, question: str, response: str) -> str:
        return stable_json_hash({"session_id": session_id, "question": question, "response": response})

    def submit(self, *, session_id: str, question: str, response: str, thumb: int,
               policy: PolicyInfo, source_helpful: bool | None = None, reason: str = "") -> bool:
        if thumb not in (-1, 1):
            raise ValueError("thumb must be -1 or 1")
        key = self.make_key(session_id, question, response)
        qh = hashlib.sha256(question.encode("utf-8")).hexdigest()
        rh = hashlib.sha256(response.encode("utf-8")).hexdigest()
        clean_reason = scrub_pii(reason)[:1000]
        try:
            with self._conn() as con:
                con.execute(
                    """
                    INSERT INTO feedback_log(
                        feedback_key, session_id, question_hash, response_hash, thumb,
                        source_helpful, reason, policy_name, policy_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (key, session_id, qh, rh, thumb,
                     None if source_helpful is None else int(source_helpful), clean_reason,
                     policy.name, policy.version),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def count(self) -> int:
        with self._conn() as con:
            return int(con.execute("SELECT COUNT(*) FROM feedback_log").fetchone()[0])

    def rows(self) -> list[dict[str, Any]]:
        with self._conn() as con:
            con.row_factory = sqlite3.Row
            return [dict(r) for r in con.execute("SELECT * FROM feedback_log ORDER BY id")]


def answer_to_ui_payload(answer, policy: PolicyInfo) -> dict[str, Any]:
    return {
        "question": answer.question,
        "answer": answer.text,
        "fallback": bool(answer.fallback),
        "citations": list(answer.citations),
        "sources": [
            {
                "chunk_id": h.chunk_id,
                "doc_id": h.doc_id,
                "score": float(h.score),
                "text": h.text,
                "source_type": str(h.metadata.get("source_type", "rag")),
                "title": str(h.metadata.get("title", "")),
                "url": str(h.metadata.get("url", "")),
                "provider": str(h.metadata.get("provider", "")),
                "model": str(h.metadata.get("model", "")),
                "externally_verified": h.metadata.get("externally_verified", h.metadata.get("source_type") != "model_internal"),
            }
            for h in answer.hits
        ],
        "policy": asdict(policy),
    }

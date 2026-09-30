from __future__ import annotations
import sqlite3, json
from pathlib import Path
from .base import ExtensionHealth

class SessionPreferencesExtension:
    name="session_preferences"
    def __init__(self, db_path: str|Path, enabled: bool=False):
        self.enabled=enabled; self.db_path=Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as c:
            c.execute("CREATE TABLE IF NOT EXISTS session_pref(session_id TEXT PRIMARY KEY, prefs_json TEXT NOT NULL)")
    def health(self): return ExtensionHealth(self.name,self.enabled,True,"session-scoped, deletable preferences")
    def set(self, session_id:str, prefs:dict):
        with sqlite3.connect(self.db_path) as c:
            c.execute("INSERT INTO session_pref VALUES(?,?) ON CONFLICT(session_id) DO UPDATE SET prefs_json=excluded.prefs_json",(session_id,json.dumps(prefs,ensure_ascii=False)))
    def get(self, session_id:str)->dict:
        with sqlite3.connect(self.db_path) as c:
            r=c.execute("SELECT prefs_json FROM session_pref WHERE session_id=?",(session_id,)).fetchone()
        return json.loads(r[0]) if r else {}
    def delete(self, session_id:str):
        with sqlite3.connect(self.db_path) as c: c.execute("DELETE FROM session_pref WHERE session_id=?",(session_id,))
    def apply(self,payload): return payload

from __future__ import annotations
import sqlite3, time
from pathlib import Path
from .base import ExtensionHealth

class LongTermMemoryExtension:
    name="long_term_memory"
    def __init__(self, db_path:str|Path, enabled:bool=False):
        self.enabled=enabled; self.db_path=Path(db_path); self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as c:
            c.execute("CREATE TABLE IF NOT EXISTS memory(user_id TEXT, memory_key TEXT, memory_value TEXT, created_at REAL, PRIMARY KEY(user_id,memory_key))")
    def health(self): return ExtensionHealth(self.name,self.enabled,True,"explicit-consent memory store")
    def put(self,user_id,key,value,consent:bool=False):
        if not consent: raise PermissionError("explicit consent required")
        with sqlite3.connect(self.db_path) as c:
            c.execute("INSERT INTO memory VALUES(?,?,?,?) ON CONFLICT(user_id,memory_key) DO UPDATE SET memory_value=excluded.memory_value, created_at=excluded.created_at",(user_id,key,value,time.time()))
    def get_all(self,user_id):
        with sqlite3.connect(self.db_path) as c:
            return c.execute("SELECT memory_key,memory_value FROM memory WHERE user_id=?",(user_id,)).fetchall()
    def delete_user(self,user_id):
        with sqlite3.connect(self.db_path) as c: c.execute("DELETE FROM memory WHERE user_id=?",(user_id,))
    def apply(self,payload): return payload

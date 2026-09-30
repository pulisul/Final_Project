from __future__ import annotations
import sqlite3, time
from pathlib import Path
from .models import Document, Chunk

SCHEMA="""
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS documents(doc_id TEXT PRIMARY KEY,path TEXT UNIQUE NOT NULL,sha256 TEXT NOT NULL,bytes INTEGER NOT NULL,indexed_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS chunks(chunk_id TEXT PRIMARY KEY,doc_id TEXT NOT NULL,ordinal INTEGER NOT NULL,strategy TEXT NOT NULL,text_sha256 TEXT NOT NULL,FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY,kind TEXT NOT NULL,status TEXT NOT NULL,started_at REAL NOT NULL,finished_at REAL,config_hash TEXT,details_json TEXT);
"""
class MetadataDB:
    def __init__(self,path:Path):
        path.parent.mkdir(parents=True,exist_ok=True); self.path=path
        with sqlite3.connect(path) as c: c.executescript(SCHEMA)
    def get_by_path(self,path:str):
        with sqlite3.connect(self.path) as c:
            return c.execute("SELECT doc_id,sha256 FROM documents WHERE path=?",(path,)).fetchone()
    def upsert_document(self,doc:Document,chunks:list[Chunk]):
        from .hashing import sha256_text
        with sqlite3.connect(self.path) as c:
            c.execute("PRAGMA foreign_keys=ON")
            c.execute("INSERT INTO documents(doc_id,path,sha256,bytes,indexed_at) VALUES(?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET doc_id=excluded.doc_id,sha256=excluded.sha256,bytes=excluded.bytes,indexed_at=excluded.indexed_at",(doc.doc_id,str(doc.path),doc.sha256,int(doc.metadata.get('bytes',0)),time.time()))
            c.execute("DELETE FROM chunks WHERE doc_id=?",(doc.doc_id,))
            c.executemany("INSERT OR REPLACE INTO chunks(chunk_id,doc_id,ordinal,strategy,text_sha256) VALUES(?,?,?,?,?)",[(x.chunk_id,x.doc_id,x.ordinal,x.strategy,sha256_text(x.text)) for x in chunks])
    def chunk_ids(self)->set[str]:
        with sqlite3.connect(self.path) as c: return {r[0] for r in c.execute("SELECT chunk_id FROM chunks")}

from __future__ import annotations
import json, time
from pathlib import Path
from .chunking import chunk_document
from .models import Document

class Indexer:
    def __init__(self, loader, embedder, store, metadata_db, cfg):
        self.loader=loader; self.embedder=embedder; self.store=store; self.db=metadata_db; self.cfg=cfg
    def index_all(self, checkpoint_path: Path) -> dict:
        checkpoint_path.parent.mkdir(parents=True,exist_ok=True)
        state={"completed":{},"failed":{},"started_at":time.time()}
        if checkpoint_path.exists():
            try: state=json.loads(checkpoint_path.read_text(encoding="utf-8"))
            except Exception: pass
        ephemeral_empty = hasattr(self.store, "rows") and not getattr(self.store, "rows")
        for path in self.loader.discover():
            key=str(path.resolve())
            try:
                doc=self.loader.load(path)
                if (not ephemeral_empty) and state.get("completed",{}).get(key)==doc.sha256:
                    continue
                old=self.db.get_by_path(key)
                if (not ephemeral_empty) and old and old[1]==doc.sha256:
                    state.setdefault("completed",{})[key]=doc.sha256; checkpoint_path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8"); continue
                chunks=chunk_document(doc,self.cfg.semantic_min_chars,self.cfg.semantic_max_chars,self.cfg.fallback_chunk_chars,self.cfg.fallback_overlap_chars)
                vectors=self.embedder.embed([c.text for c in chunks])
                if any((not v) or any((x!=x) for x in v) or sum(abs(x) for x in v)==0 for v in vectors): raise ValueError("invalid embedding vector")
                if old and old[0] != doc.doc_id: self.store.delete_doc(old[0])
                self.store.upsert(chunks,vectors); self.db.upsert_document(doc,chunks)
                state.setdefault("completed",{})[key]=doc.sha256; state.get("failed",{}).pop(key,None)
            except Exception as e:
                state.setdefault("failed",{})[key]=f"{type(e).__name__}: {e}"
            checkpoint_path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
        state["finished_at"]=time.time(); checkpoint_path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
        total=len(state.get("completed",{}))+len(state.get("failed",{})); ok=len(state.get("completed",{}))
        state["success_ratio"]=(ok/total if total else 1.0)
        return state

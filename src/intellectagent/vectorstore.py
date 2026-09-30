from __future__ import annotations
import math
from typing import Protocol, Sequence
from .models import Chunk, SearchHit

class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...
    def query(self, vector: Sequence[float], top_k: int) -> list[SearchHit]: ...
    def delete_doc(self, doc_id: str) -> None: ...

class InMemoryVectorStore:
    def __init__(self): self.rows: dict[str, tuple[Chunk, list[float]]] = {}
    def upsert(self, chunks, vectors):
        for c,v in zip(chunks,vectors,strict=True): self.rows[c.chunk_id]=(c,list(v))
    def delete_doc(self, doc_id):
        for cid in [cid for cid,(c,_) in self.rows.items() if c.doc_id==doc_id]: self.rows.pop(cid,None)
    def query(self, vector, top_k):
        q=list(vector)
        def cos(a,b):
            den=(math.sqrt(sum(x*x for x in a))*math.sqrt(sum(x*x for x in b))) or 1.0
            return sum(x*y for x,y in zip(a,b))/den
        ranked=sorted(((cos(q,v),c) for c,v in self.rows.values()), key=lambda x:x[0], reverse=True)[:top_k]
        return [SearchHit(c.chunk_id,c.doc_id,c.text,float(s),dict(c.metadata)) for s,c in ranked]

class ChromaVectorStore:
    def __init__(self, path: str, collection_name: str):
        import chromadb
        self.client=chromadb.PersistentClient(path=path)
        self.collection=self.client.get_or_create_collection(name=collection_name, metadata={"hnsw:space":"cosine"})
    def upsert(self, chunks, vectors):
        if not chunks: return
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks], embeddings=[list(map(float,v)) for v in vectors],
            documents=[c.text for c in chunks],
            metadatas=[{"doc_id":c.doc_id,"strategy":c.strategy,"ordinal":c.ordinal,**{k:str(v) for k,v in c.metadata.items()}} for c in chunks])
    def delete_doc(self, doc_id): self.collection.delete(where={"doc_id":doc_id})
    def chunk_ids(self) -> set[str]:
        r=self.collection.get(include=[])
        return set(r.get("ids") or [])
    def query(self, vector, top_k):
        r=self.collection.query(query_embeddings=[list(map(float,vector))], n_results=top_k, include=["documents","metadatas","distances"])
        out=[]
        ids=(r.get("ids") or [[]])[0]; docs=(r.get("documents") or [[]])[0]; mets=(r.get("metadatas") or [[]])[0]; dists=(r.get("distances") or [[]])[0]
        for cid,text,meta,dist in zip(ids,docs,mets,dists,strict=False):
            meta=meta or {}; score=1.0-float(dist)
            out.append(SearchHit(cid,str(meta.get("doc_id","")),text or "",score,dict(meta)))
        return out

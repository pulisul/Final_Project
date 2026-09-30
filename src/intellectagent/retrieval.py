from __future__ import annotations
from typing import Protocol
from .models import SearchHit

class Reranker(Protocol):
    def rerank(self, query:str, hits:list[SearchHit]) -> list[SearchHit]: ...

class NoopReranker:
    def rerank(self,query,hits): return hits

class CrossEncoderReranker:
    def __init__(self, model_name:str, device: str | None = None):
        from sentence_transformers import CrossEncoder
        self.model=CrossEncoder(model_name, device=device) if device else CrossEncoder(model_name)
        self.device=device
    def rerank(self, query, hits):
        if not hits:return []
        scores=self.model.predict([(query,h.text) for h in hits])
        merged=[SearchHit(h.chunk_id,h.doc_id,h.text,float(s),h.metadata) for h,s in zip(hits,scores,strict=True)]
        return sorted(merged,key=lambda h:h.score,reverse=True)

class Retriever:
    def __init__(self,embedder,store,reranker=None): self.embedder=embedder; self.store=store; self.reranker=reranker or NoopReranker()
    def search(self,query:str,top_k:int,candidate_k:int|None=None):
        qv=self.embedder.embed([query])[0]
        hits=self.store.query(qv,candidate_k or top_k)
        try:
            ranked=self.reranker.rerank(query,hits)
        except Exception:
            # Reranker is an enhancement, never a single point of failure.
            ranked=hits
        return ranked[:top_k]

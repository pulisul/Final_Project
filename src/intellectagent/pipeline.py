from __future__ import annotations
from .grounding import validate_grounding
from .models import Answer

class CoreRAG:
    def __init__(self,retriever,generator,top_k:int=5,fallback_min_score:float=0.12,extensions=None,*,candidate_k:int|None=None,runtime_profile=None):
        self.retriever=retriever; self.generator=generator; self.top_k=top_k; self.fallback_min_score=fallback_min_score
        self.extensions=extensions; self.candidate_k=candidate_k; self.runtime_profile=runtime_profile
    def ask(self,question:str)->Answer:
        hits=self.retriever.search(question,self.top_k,candidate_k=self.candidate_k)
        if not hits or hits[0].score < self.fallback_min_score:
            return Answer(question,"관련 근거 문서를 찾지 못했습니다. 질문을 더 구체적으로 바꿔 주세요.",(),tuple(hits),True)
        text=self.generator.generate(question,hits)
        g=validate_grounding(text,hits)
        if not g.ok:
            return Answer(question,"근거 검증을 통과하지 못해 답변을 제공하지 않습니다. 질문을 더 구체적으로 바꿔 주세요.",(),tuple(hits),True)
        answer=Answer(question,text,g.cited_ids,tuple(hits),False)
        if self.extensions is not None:
            answer=self.extensions.apply_if_ready("grounding_verifier",answer)
        return answer

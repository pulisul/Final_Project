from __future__ import annotations
from .gating import gate_score
from .grounding import validate_grounding
from .models import Answer
from .textnorm import clean_text

class CoreRAG:
    def __init__(self,retriever,generator,top_k:int=5,fallback_min_score:float=0.12,extensions=None,*,candidate_k:int|None=None,runtime_profile=None,grounding_check_numbers:bool=False):
        self.retriever=retriever; self.generator=generator; self.top_k=top_k; self.fallback_min_score=fallback_min_score
        self.extensions=extensions; self.candidate_k=candidate_k; self.runtime_profile=runtime_profile
        self.grounding_check_numbers=grounding_check_numbers
    @staticmethod
    def _gate(hits)->float:
        return gate_score(hits)  # 워크플로(orchestration)와 같은 규칙을 쓰도록 gating.py 에 둔다
    def ask(self,question:str)->Answer:
        question=clean_text(question)   # NFD·제로폭 문자가 섞인 질문도 색인된 본문(같은 clean_text)과 맞는다
        hits=self.retriever.search(question,self.top_k,candidate_k=self.candidate_k)
        if not hits or self._gate(hits) < self.fallback_min_score:
            return Answer(question,"관련 근거 문서를 찾지 못했습니다. 질문을 더 구체적으로 바꿔 주세요.",(),tuple(hits),True)
        text=self.generator.generate(question,hits)
        g=validate_grounding(text,hits,check_numbers=self.grounding_check_numbers)
        if g.model_refused:   # 모델이 스스로 '근거 없음'이라 답함 → 점수 관문 거절과 같은 문구·같은 처리(형식 위반으로 세지 않는다)
            return Answer(question,"관련 근거 문서를 찾지 못했습니다. 질문을 더 구체적으로 바꿔 주세요.",(),tuple(hits),True)
        if not g.ok:
            return Answer(question,"근거 검증을 통과하지 못해 답변을 제공하지 않습니다. 질문을 더 구체적으로 바꿔 주세요.",(),tuple(hits),True)
        answer=Answer(question,text,g.cited_ids,tuple(hits),False)
        if self.extensions is not None:
            answer=self.extensions.apply_if_ready("grounding_verifier",answer)
        return answer

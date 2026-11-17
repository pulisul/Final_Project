"""질의 확장: HyDE (AutoRAG ``queryexpansion/hyde.py``; Gao et al. 2022 "Precise Zero-shot Dense Retrieval without Relevance Labels").

LLM 이 질문에 답하는 *가상의 문단* 을 쓰고, 그 문단으로 검색한다. 질문과 문서의 어휘/문체가 달라 벡터 검색이 빗나갈 때 시도한다.
주의: 가상 문단은 환각일 수 있다. 검색 *입력* 으로만 쓰고 답변 근거로는 절대 쓰지 않는다(근거는 검색된 실제 청크).
원본과 같이 기본은 가상 문단만으로 검색하고(``include_original=False``), LLM 실패/빈 응답이면 원 질문으로 되돌아간다.
"""

from __future__ import annotations

import re
from typing import Callable

from .hybrid import rrf_fuse
from .models import SearchHit

LLM = Callable[[str], str]
HYDE_PROMPT = "다음 질문에 답하는 짧은 문단을 한국어로 작성하세요."  # 원본: "Please write a passage to answer the question"
_THINK = re.compile(r"<think>.*?</think>", re.S)


def hyde_passage(llm: LLM, query: str, prompt: str = HYDE_PROMPT) -> str:
    try:
        return _THINK.sub("", llm(f"{prompt}\nQuestion: {query}\nPassage:")).strip()  # 원본 프롬프트 형식 그대로
    except Exception:
        return ""


class HydeRetriever:
    def __init__(self, inner, llm: LLM, prompt: str = HYDE_PROMPT, include_original: bool = False, rrf_k: int = 60):
        self.inner, self.llm, self.prompt = inner, llm, prompt
        self.include_original, self.rrf_k = include_original, rrf_k
        self.last_passage = ""  # 디버깅/보고용: 마지막으로 쓴 가상 문단

    def search(self, query: str, top_k: int, *args, **kwargs) -> list[SearchHit]:
        passage = hyde_passage(self.llm, query, self.prompt)
        self.last_passage = passage
        if not passage:
            return list(self.inner.search(query, top_k, *args, **kwargs))
        hyde_hits = list(self.inner.search(passage, top_k, *args, **kwargs))
        if not self.include_original:
            return hyde_hits
        orig_hits = list(self.inner.search(query, top_k, *args, **kwargs))
        if not hyde_hits or not orig_hits:
            return (hyde_hits or orig_hits)[:top_k]
        return rrf_fuse([orig_hits, hyde_hits], top_k=top_k, k=self.rrf_k)

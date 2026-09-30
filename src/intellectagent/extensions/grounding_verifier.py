from __future__ import annotations
import re
from dataclasses import replace
from .base import ExtensionHealth

class GroundingVerifierExtension:
    name = "grounding_verifier"
    def __init__(self, enabled: bool = False, min_token_overlap: float = 0.18):
        self.enabled = enabled
        self.min_token_overlap = min_token_overlap

    def health(self):
        return ExtensionHealth(self.name, self.enabled, True, "lexical sentence-to-evidence verification")

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {t.lower() for t in re.findall(r"[A-Za-z0-9가-힣]{2,}", text)}

    def apply(self, answer):
        if getattr(answer, "fallback", False):
            return answer
        evidence = " ".join(getattr(h, "text", "") for h in answer.hits)
        e = self._tokens(evidence)
        if not e:
            return replace(answer, text="근거 검증에 실패했습니다.", citations=(), fallback=True)
        unsupported=[]
        clean_text=re.sub(r"\[[^\]]+\]", " ", answer.text)
        for s in re.split(r"(?<=[.!?。])\s+", clean_text.strip()):
            q=self._tokens(s)
            if not q: continue
            if len(q & e) / max(1, len(q)) < self.min_token_overlap:
                unsupported.append(s)
        if unsupported:
            return replace(answer, text="근거 검증을 통과하지 못해 답변을 제공하지 않습니다.", citations=(), fallback=True)
        return answer

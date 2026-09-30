from __future__ import annotations
import re
from typing import Any, Protocol
from .models import SearchHit

class Generator(Protocol):
    def generate(self, question:str, hits:list[SearchHit]) -> str: ...

class ExtractiveGenerator:
    """Safe deterministic baseline: never invents beyond retrieved snippets."""
    def generate(self, question, hits):
        if not hits:return "관련 근거 문서를 찾지 못했습니다. 질문을 더 구체적으로 바꿔 주세요."
        parts=[]
        for h in hits[:3]:
            sent=re.split(r"(?<=[.!?다요])\s+|\n+",h.text.strip())[0].strip()
            if sent: parts.append(f"{sent} [{h.chunk_id}]")
        return "\n".join(parts) if parts else "관련 근거 문서를 찾지 못했습니다."


def build_grounded_prompt(question: str, hits: list[SearchHit]) -> str:
    context="\n\n".join(f"SOURCE_ID={h.chunk_id}\n{h.text}" for h in hits)
    return (
        "당신은 문서 근거 기반 질의응답기입니다. 아래 CONTEXT만 사용하세요. "
        "외부 지식이나 추측을 추가하지 마세요. 각 사실 주장 문장 끝에 반드시 [SOURCE_ID] 형식으로 "
        "실제 SOURCE_ID를 붙이세요. 근거가 부족하면 정확히 '관련 근거 문서를 찾지 못했습니다'라고 답하세요.\n\n"
        f"QUESTION:\n{question}\n\nCONTEXT:\n{context}"
    )


class OllamaGenerator:
    def __init__(self,model:str, client: Any | None = None):
        self.model=model
        self._client=client
    def _client_or_module(self):
        if self._client is not None:
            return self._client
        import ollama
        return ollama
    def generate(self,question,hits):
        if not hits:
            return "관련 근거 문서를 찾지 못했습니다. 질문을 더 구체적으로 바꿔 주세요."
        client=self._client_or_module()
        prompt=build_grounded_prompt(question,hits)
        r=client.chat(model=self.model,messages=[{"role":"user","content":prompt}])
        if isinstance(r, dict):
            return str((r.get("message") or {}).get("content", ""))
        message=getattr(r,"message",None)
        content=getattr(message,"content","") if message is not None else ""
        return str(content)

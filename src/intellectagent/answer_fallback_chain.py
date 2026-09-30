from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import Answer
from .ollama_knowledge import OllamaKnowledgeEngine, OllamaKnowledgeError
from .web_fallback import WebFallbackEngine, WebSearchError


@dataclass(frozen=True)
class FallbackDecision:
    source: str
    reason: str


class EmptyRAGFallbackChain:
    """Priority chain used only when there are zero local RAG documents.

    Priority:
      1) web-search evidence
      2) Ollama pretrained/internal knowledge
    """

    def __init__(self, web_engine: WebFallbackEngine | None, knowledge_engine: OllamaKnowledgeEngine | None):
        self.web_engine = web_engine
        self.knowledge_engine = knowledge_engine
        self.last_decision: FallbackDecision | None = None

    def ask(self, question: str) -> Answer:
        web_failure_reason = "web_disabled"
        if self.web_engine is not None:
            try:
                web_answer = self.web_engine.ask(question)
                if not web_answer.fallback:
                    self.last_decision = FallbackDecision("web", "web_grounded_answer")
                    return web_answer
                web_failure_reason = "web_no_usable_grounded_evidence"
            except WebSearchError:
                web_failure_reason = "web_search_unavailable"
            except Exception:
                # Search/network/provider failures should not prevent the local model
                # from serving the explicit final fallback path.
                web_failure_reason = "web_search_error"

        if self.knowledge_engine is not None:
            try:
                answer = self.knowledge_engine.ask(question, reason=web_failure_reason)
                self.last_decision = FallbackDecision("ollama_internal", web_failure_reason)
                return answer
            except OllamaKnowledgeError:
                pass

        self.last_decision = FallbackDecision("none", f"{web_failure_reason};ollama_unavailable")
        return Answer(
            question=question,
            text=(
                "RAG 문서가 없고 인터넷 검색을 사용할 수 없으며 Ollama 내부지식에도 연결하지 못했습니다. "
                "네트워크 또는 Ollama 실행 상태를 확인해 주세요."
            ),
            citations=(),
            hits=(),
            fallback=True,
            policy="empty-rag-no-source",
        )

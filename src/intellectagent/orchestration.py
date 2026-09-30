from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .grounding import GroundingResult, validate_grounding
from .models import Answer, SearchHit

FALLBACK_NO_EVIDENCE = "관련 근거 문서를 찾지 못했습니다. 질문을 더 구체적으로 바꿔 주세요."
FALLBACK_UNGROUNDED = "근거 검증을 통과하지 못해 답변을 제공하지 않습니다. 질문을 더 구체적으로 바꿔 주세요."


@dataclass
class WorkflowState:
    question: str
    hits: list[SearchHit] = field(default_factory=list)
    generated_text: str = ""
    grounding: GroundingResult | None = None
    route: str = "start"
    answer: Answer | None = None
    trace: list[str] = field(default_factory=list)


class AgenticRAGWorkflow:
    """Dependency-light orchestration used by tests and as LangGraph node logic.

    The core invariant is that fallback is a deterministic safety rule, not an LLM/RL action.
    """

    def __init__(self, retriever: Any, generator: Any, *, top_k: int = 5, candidate_k: int | None = None,
                 fallback_min_score: float = 0.12):
        self.retriever = retriever
        self.generator = generator
        self.top_k = int(top_k)
        self.candidate_k = candidate_k
        self.fallback_min_score = float(fallback_min_score)

    def retrieve_node(self, state: WorkflowState) -> WorkflowState:
        state.trace.append("retrieve")
        state.hits = list(self.retriever.search(state.question, self.top_k, candidate_k=self.candidate_k))
        return state

    def route_after_retrieve(self, state: WorkflowState) -> str:
        if not state.hits:
            state.route = "fallback_no_evidence"
        elif state.hits[0].score < self.fallback_min_score:
            state.route = "fallback_no_evidence"
        else:
            state.route = "generate"
        state.trace.append(f"route:{state.route}")
        return state.route

    def generate_node(self, state: WorkflowState) -> WorkflowState:
        state.trace.append("generate")
        state.generated_text = self.generator.generate(state.question, state.hits)
        return state

    def grounding_node(self, state: WorkflowState) -> WorkflowState:
        state.trace.append("grounding")
        state.grounding = validate_grounding(state.generated_text, state.hits)
        return state

    def route_after_grounding(self, state: WorkflowState) -> str:
        state.route = "finalize" if state.grounding and state.grounding.ok else "fallback_ungrounded"
        state.trace.append(f"route:{state.route}")
        return state.route

    def finalize_node(self, state: WorkflowState) -> WorkflowState:
        assert state.grounding is not None
        state.trace.append("finalize")
        state.answer = Answer(
            question=state.question,
            text=state.generated_text,
            citations=state.grounding.cited_ids,
            hits=tuple(state.hits),
            fallback=False,
        )
        return state

    def fallback_node(self, state: WorkflowState, reason: str) -> WorkflowState:
        state.trace.append(reason)
        text = FALLBACK_UNGROUNDED if reason == "fallback_ungrounded" else FALLBACK_NO_EVIDENCE
        state.answer = Answer(state.question, text, (), tuple(state.hits), True)
        return state

    def run(self, question: str) -> tuple[Answer, WorkflowState]:
        state = WorkflowState(question=question)
        self.retrieve_node(state)
        route = self.route_after_retrieve(state)
        if route == "fallback_no_evidence":
            self.fallback_node(state, route)
            return state.answer, state  # type: ignore[return-value]

        self.generate_node(state)
        self.grounding_node(state)
        route = self.route_after_grounding(state)
        if route == "fallback_ungrounded":
            self.fallback_node(state, route)
        else:
            self.finalize_node(state)
        return state.answer, state  # type: ignore[return-value]

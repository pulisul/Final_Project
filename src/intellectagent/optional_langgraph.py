"""Optional LangGraph adapter.

The pure-python AgenticRAGWorkflow is authoritative. LangGraph only visualizes/executes
that same node contract, so an unavailable LangGraph package never breaks Core RAG.
"""
from __future__ import annotations

from typing import Any, TypedDict

from .orchestration import AgenticRAGWorkflow, WorkflowState


class GraphState(TypedDict, total=False):
    question: str
    workflow_state: WorkflowState
    answer: Any


def build_graph(core_rag=None, *, workflow: AgenticRAGWorkflow | None = None):
    from langgraph.graph import END, START, StateGraph

    if workflow is None:
        if core_rag is None:
            raise ValueError("core_rag or workflow is required")
        workflow = AgenticRAGWorkflow(
            core_rag.retriever,
            core_rag.generator,
            top_k=core_rag.top_k,
            candidate_k=getattr(core_rag, "candidate_k", None),
            fallback_min_score=core_rag.fallback_min_score,
        )

    def retrieve(state: GraphState):
        ws = WorkflowState(question=state["question"])
        workflow.retrieve_node(ws)
        return {"workflow_state": ws}

    def route_retrieve(state: GraphState):
        return workflow.route_after_retrieve(state["workflow_state"])

    def generate(state: GraphState):
        ws = state["workflow_state"]
        workflow.generate_node(ws)
        return {"workflow_state": ws}

    def grounding(state: GraphState):
        ws = state["workflow_state"]
        workflow.grounding_node(ws)
        return {"workflow_state": ws}

    def route_grounding(state: GraphState):
        return workflow.route_after_grounding(state["workflow_state"])

    def finalize(state: GraphState):
        ws = state["workflow_state"]
        workflow.finalize_node(ws)
        return {"workflow_state": ws, "answer": ws.answer}

    def fallback_no_evidence(state: GraphState):
        ws = state["workflow_state"]
        workflow.fallback_node(ws, "fallback_no_evidence")
        return {"workflow_state": ws, "answer": ws.answer}

    def fallback_ungrounded(state: GraphState):
        ws = state["workflow_state"]
        workflow.fallback_node(ws, "fallback_ungrounded")
        return {"workflow_state": ws, "answer": ws.answer}

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("grounding", grounding)
    graph.add_node("finalize", finalize)
    graph.add_node("fallback_no_evidence", fallback_no_evidence)
    graph.add_node("fallback_ungrounded", fallback_ungrounded)

    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges("retrieve", route_retrieve, {
        "generate": "generate",
        "fallback_no_evidence": "fallback_no_evidence",
    })
    graph.add_edge("generate", "grounding")
    graph.add_conditional_edges("grounding", route_grounding, {
        "finalize": "finalize",
        "fallback_ungrounded": "fallback_ungrounded",
    })
    graph.add_edge("finalize", END)
    graph.add_edge("fallback_no_evidence", END)
    graph.add_edge("fallback_ungrounded", END)
    return graph.compile()

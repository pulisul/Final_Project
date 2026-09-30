from intellectagent.models import SearchHit
from intellectagent.orchestration import AgenticRAGWorkflow


class RetrieverStub:
    def __init__(self, hits): self.hits=hits
    def search(self, q, top_k, candidate_k=None): return self.hits[:top_k]


class GoodGenerator:
    def generate(self,q,hits): return f"사과는 과일입니다. [{hits[0].chunk_id}]"


class BadGenerator:
    def generate(self,q,hits): return "근거 없는 답변입니다."


def hit(score=0.9):
    return SearchHit("a"*20,"doc1","사과는 붉은 과일입니다.",score,{})


def test_workflow_success_path():
    wf=AgenticRAGWorkflow(RetrieverStub([hit()]),GoodGenerator(),top_k=1,fallback_min_score=0.1)
    answer,state=wf.run("사과")
    assert not answer.fallback
    assert answer.citations == ("a"*20,)
    assert state.trace == ["retrieve","route:generate","generate","grounding","route:finalize","finalize"]


def test_workflow_low_score_fallback_path():
    wf=AgenticRAGWorkflow(RetrieverStub([hit(0.01)]),GoodGenerator(),top_k=1,fallback_min_score=0.1)
    answer,state=wf.run("사과")
    assert answer.fallback
    assert "generate" not in state.trace
    assert "fallback_no_evidence" in state.trace


def test_workflow_ungrounded_fallback_path():
    wf=AgenticRAGWorkflow(RetrieverStub([hit()]),BadGenerator(),top_k=1,fallback_min_score=0.1)
    answer,state=wf.run("사과")
    assert answer.fallback
    assert "grounding" in state.trace
    assert "fallback_ungrounded" in state.trace

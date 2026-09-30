from intellectagent.answer_fallback_chain import EmptyRAGFallbackChain
from intellectagent.models import Answer, SearchHit
from intellectagent.ollama_knowledge import OllamaKnowledgeEngine
from intellectagent.web_fallback import WebSearchError
from intellectagent.ui_support import PolicyInfo, answer_to_ui_payload


class WebSuccess:
    def ask(self, question):
        hit = SearchHit(
            chunk_id="abc123abc123abc123ab",
            doc_id="https://example.com",
            text="웹 근거",
            score=1.0,
            metadata={"source_type": "web", "title": "웹", "url": "https://example.com"},
        )
        return Answer(question, "웹 답변 [abc123abc123abc123ab]", (hit.chunk_id,), (hit,), False, policy="web-fallback")


class WebFailure:
    def ask(self, question):
        raise WebSearchError("offline")


class WebNoEvidence:
    def ask(self, question):
        return Answer(question, "웹 근거 없음", (), (), True, policy="web-fallback")


class FakeKnowledge:
    def __init__(self):
        self.calls = 0
    def ask(self, question, *, reason=""):
        self.calls += 1
        hit = SearchHit(
            chunk_id="modelsource1234567890",
            doc_id="ollama://qwen-test",
            text="모델 내부지식이며 외부 검증되지 않음",
            score=0.0,
            metadata={
                "source_type": "model_internal",
                "title": "Ollama 내부지식 · qwen-test",
                "url": "",
                "provider": "ollama",
                "model": "qwen-test",
                "externally_verified": False,
            },
        )
        return Answer(question, "내부지식 답변", (), (hit,), False, policy=f"ollama-internal:{reason}")


class FakeOllamaClient:
    def chat(self, **kwargs):
        return {"message": {"content": "모델 내부지식으로 답변합니다."}}


def test_web_has_priority_over_ollama_internal():
    knowledge = FakeKnowledge()
    chain = EmptyRAGFallbackChain(WebSuccess(), knowledge)
    answer = chain.ask("질문")
    assert answer.policy == "web-fallback"
    assert knowledge.calls == 0
    assert chain.last_decision.source == "web"


def test_web_network_failure_falls_back_to_ollama_internal():
    knowledge = FakeKnowledge()
    chain = EmptyRAGFallbackChain(WebFailure(), knowledge)
    answer = chain.ask("질문")
    assert answer.policy.startswith("ollama-internal:")
    assert knowledge.calls == 1
    assert chain.last_decision.source == "ollama_internal"
    assert chain.last_decision.reason == "web_search_unavailable"


def test_web_no_grounded_evidence_falls_back_to_ollama_internal():
    knowledge = FakeKnowledge()
    chain = EmptyRAGFallbackChain(WebNoEvidence(), knowledge)
    answer = chain.ask("질문")
    assert answer.policy.startswith("ollama-internal:")
    assert knowledge.calls == 1
    assert chain.last_decision.reason == "web_no_usable_grounded_evidence"


def test_ollama_internal_provenance_is_explicit_and_has_no_fake_url():
    engine = OllamaKnowledgeEngine("qwen-test", client=FakeOllamaClient())
    answer = engine.ask("RAG란?")
    assert answer.fallback is False
    assert answer.citations == ()
    assert len(answer.hits) == 1
    hit = answer.hits[0]
    assert hit.metadata["source_type"] == "model_internal"
    assert hit.metadata["externally_verified"] is False
    assert hit.metadata["url"] == ""
    payload = answer_to_ui_payload(answer, PolicyInfo("rule", "v1", False, ""))
    src = payload["sources"][0]
    assert src["source_type"] == "model_internal"
    assert src["externally_verified"] is False
    assert src["url"] == ""


def test_production_config_enables_internal_knowledge_fallback():
    import json
    from pathlib import Path
    flags = json.loads(Path("config/core.production.json").read_text())["feature_flags"]
    assert flags["use_empty_rag_web_fallback"] is True
    assert flags["use_ollama_knowledge_fallback"] is True

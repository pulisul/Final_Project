from intellectagent.models import SearchHit
from intellectagent.web_fallback import WebFallbackEngine, DDGSWebSearchProvider, settings_from_feature_flags
from intellectagent.ui_support import PolicyInfo, answer_to_ui_payload


class FakeProvider:
    def search(self, query: str, *, max_results: int = 5):
        return [
            SearchHit(
                chunk_id="0123456789abcdefabcd",
                doc_id="https://example.com/rag",
                text="RAG는 검색된 근거를 생성 모델에 제공하는 방식입니다.",
                score=1.0,
                metadata={"source_type": "web", "title": "RAG 소개", "url": "https://example.com/rag"},
            )
        ]


class GroundedFakeGenerator:
    def generate(self, question, hits):
        return "RAG는 검색된 근거를 생성 모델에 제공하는 방식입니다. [0123456789abcdefabcd]"


class UngroundedFakeGenerator:
    def generate(self, question, hits):
        return "출처 표시가 없는 답변입니다."


def test_web_fallback_grounded_answer_and_source_metadata():
    engine = WebFallbackEngine(FakeProvider(), GroundedFakeGenerator(), max_results=5)
    answer = engine.ask("RAG란?")
    assert answer.fallback is False
    assert answer.policy == "web-fallback"
    assert answer.citations == ("0123456789abcdefabcd",)
    payload = answer_to_ui_payload(answer, PolicyInfo("rule", "v1", False, ""))
    assert payload["sources"][0]["source_type"] == "web"
    assert payload["sources"][0]["url"] == "https://example.com/rag"
    assert payload["sources"][0]["title"] == "RAG 소개"


def test_web_fallback_rejects_ungrounded_generation():
    engine = WebFallbackEngine(FakeProvider(), UngroundedFakeGenerator())
    answer = engine.ask("RAG란?")
    assert answer.fallback is True
    assert "근거 검증" in answer.text


def test_ddgs_chunk_id_is_grounding_compatible_and_stable():
    a = DDGSWebSearchProvider._chunk_id("https://example.com", "title", "body")
    b = DDGSWebSearchProvider._chunk_id("https://example.com", "title", "body")
    assert a == b
    assert len(a) == 20
    assert all(c in "0123456789abcdef" for c in a)


def test_web_search_settings_default_off_and_configurable():
    assert settings_from_feature_flags({}).enabled is False
    s = settings_from_feature_flags({
        "use_empty_rag_web_fallback": True,
        "web_search_max_results": 7,
        "web_search_region": "kr-kr",
        "web_search_timeout_seconds": 9,
    })
    assert s.enabled is True
    assert s.max_results == 7
    assert s.region == "kr-kr"
    assert s.timeout_seconds == 9

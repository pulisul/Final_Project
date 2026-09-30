from intellectagent.generation import OllamaGenerator, build_grounded_prompt
from intellectagent.models import SearchHit


def test_prompt_contains_real_source_id_and_safety_instruction():
    h=SearchHit("b"*20,"doc","RAG는 검색과 생성을 결합합니다.",0.9,{})
    p=build_grounded_prompt("RAG란?",[h])
    assert "SOURCE_ID="+("b"*20) in p
    assert "외부 지식이나 추측을 추가하지 마세요" in p


class FakeOllama:
    def chat(self, model, messages):
        assert model == "fake-model"
        assert messages and messages[0]["role"] == "user"
        return {"message":{"content":"RAG 답변 ["+("c"*20)+"]"}}


def test_ollama_generator_is_testable_without_server():
    h=SearchHit("c"*20,"doc","RAG",0.9,{})
    g=OllamaGenerator("fake-model", client=FakeOllama())
    assert "["+("c"*20)+"]" in g.generate("RAG?",[h])

from intellectagent.embeddings import HashEmbedder
from intellectagent.vectorstore import InMemoryVectorStore
from intellectagent.retrieval import Retriever
from intellectagent.generation import ExtractiveGenerator
from intellectagent.pipeline import CoreRAG
from intellectagent.models import Chunk

def build():
    e=HashEmbedder(); s=InMemoryVectorStore(); c=Chunk("a"*20,"doc1","사과는 붉은 과일입니다.","semantic",0,{})
    s.upsert([c],e.embed([c.text])); return CoreRAG(Retriever(e,s),ExtractiveGenerator(),top_k=1,fallback_min_score=-1.0)

def test_grounded_answer_has_citation():
    a=build().ask("사과")
    assert not a.fallback and a.citations==( "a"*20, )

def test_ungrounded_generator_is_blocked():
    class Bad:
        def generate(self,q,h): return "근거 없는 답변"
    p=build(); p.generator=Bad()
    a=p.ask("사과")
    assert a.fallback and not a.citations

def test_reranker_failure_falls_back_to_base_search():
    from intellectagent.embeddings import HashEmbedder
    from intellectagent.vectorstore import InMemoryVectorStore
    from intellectagent.retrieval import Retriever
    from intellectagent.models import Chunk
    class Broken:
        def rerank(self,q,h): raise RuntimeError("boom")
    e=HashEmbedder(); s=InMemoryVectorStore(); c=Chunk("b"*20,"doc2","검색 증강 생성 RAG", "semantic",0,{})
    s.upsert([c],e.embed([c.text]))
    hits=Retriever(e,s,Broken()).search("RAG",1)
    assert hits and hits[0].doc_id=="doc2"

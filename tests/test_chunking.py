from pathlib import Path
from intellectagent.models import Document
from intellectagent.chunking import chunk_document

def test_chunks_have_stable_unique_ids():
    d=Document("doc",Path("a.md"),"# A\nhello world\n\n# B\nsecond paragraph"*30,"x")
    c1=chunk_document(d); c2=chunk_document(d)
    assert [x.chunk_id for x in c1]==[x.chunk_id for x in c2]
    assert len({x.chunk_id for x in c1})==len(c1)

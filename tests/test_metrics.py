from intellectagent.evaluation import hit_rate_at_k,mrr
from intellectagent.models import EvalCase,SearchHit

def test_metrics():
    c=EvalCase("1","q",("d2",))
    h=[SearchHit("c1","d1","",.9,{}),SearchHit("c2","d2","",.8,{})]
    assert hit_rate_at_k([(c,h)])==1.0
    assert mrr([(c,h)])==0.5

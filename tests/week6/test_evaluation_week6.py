from intellectagent.evaluation_week6 import Week6EvalCase, Week6RunConfig, evaluate_pipeline_week6
from intellectagent.models import Answer, SearchHit

class P:
    def ask(self,q):
        h=SearchHit('c1','d1','text',1.0,{})
        return Answer(q,'answer [c1]',('c1',),(h,),False)

def test_metric_groups_are_separated():
    case=Week6EvalCase('1','q',('d1',),'ref',False,'fact')
    r=evaluate_pipeline_week6(P(),[case],Week6RunConfig('r',1,'s',5,'d'))
    assert set(r['metrics'])=={'retrieval','generation','fallback','latency'}
    assert r['metrics']['retrieval']['hit_rate_at_k']==1.0
    assert r['metrics']['generation']['citation_validity']==1.0

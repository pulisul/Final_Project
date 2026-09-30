from intellectagent.ragas_adapter import run_with_injected_evaluator

def test_ragas_boundary_can_be_tested_without_ragas_package():
    rows=[{'user_input':'q'}]
    out=run_with_injected_evaluator(rows,lambda xs:{'count':len(xs),'faithfulness':1.0})
    assert out['count']==1

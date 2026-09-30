from pathlib import Path
from intellectagent.human_eval import summarize_human_eval

def test_human_eval_summary(tmp_path:Path):
    p=tmp_path/'h.csv'
    p.write_text('case_id,evaluator_id,accuracy,groundedness,question_satisfaction,source_utility,readability,comment\nq1,u1,5,4,5,4,5,ok\n',encoding='utf-8')
    r=summarize_human_eval(p)
    assert r['completed_rows']==1
    assert r['means']['accuracy']==5

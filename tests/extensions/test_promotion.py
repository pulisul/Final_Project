import json
from pathlib import Path
from intellectagent.promotion import PromotionGate

def test_promotion_blocks_regression(tmp_path:Path):
    (tmp_path/'b.json').write_text(json.dumps({'hit_rate_at_k':1,'citation_success':1,'fallback_accuracy':1}))
    (tmp_path/'p.json').write_text(json.dumps({'default':{'min_delta':{},'max_regression':{'hit_rate_at_k':0.02,'citation_success':0.0}}}))
    r=PromotionGate(tmp_path/'b.json',tmp_path/'p.json').evaluate('x',{'hit_rate_at_k':0.8,'citation_success':1,'tests_failed':0})
    assert not r.passed

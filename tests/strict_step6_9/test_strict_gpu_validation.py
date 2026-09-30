from intellectagent.strict_gpu_validation import CandidateResult, choose_candidate, finite_nonzero_vectors

def test_finite_nonzero_vectors():
    assert finite_nonzero_vectors([[1.0,0.0],[0.2,0.3]])
    assert not finite_nonzero_vectors([[0.0,0.0]])

def test_choose_candidate_quality_first_then_latency():
    xs=[CandidateResult(5,.8,.7,100,120,60,40,0),CandidateResult(10,.9,.6,200,220,80,120,0),CandidateResult(20,.9,.6,150,170,70,80,0)]
    assert choose_candidate(xs).candidate_k==20

def test_choose_candidate_smaller_k_if_all_equal():
    xs=[CandidateResult(20,.9,.8,100,120,60,40,0),CandidateResult(5,.9,.8,100,120,60,40,0)]
    assert choose_candidate(xs).candidate_k==5

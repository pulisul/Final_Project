from intellectagent.strict_services import validate_grounded_text, first_failed_stage

def test_grounded_text_accepts_real_citation():
    r=validate_grounded_text('근거입니다. [c1]',['c1'])
    assert r['pass'] and r['citation_ids_found']==['c1']

def test_grounded_text_rejects_uncited_answer():
    assert validate_grounded_text('근거 없는 답변',['c1'])['pass'] is False

def test_grounded_text_accepts_explicit_fallback():
    r=validate_grounded_text('관련 근거 문서를 찾지 못했습니다',['c1'])
    assert r['pass'] and r['fallback']

def test_first_failed_stage():
    assert first_failed_stage({'a':{'pass':True},'b':{'pass':False},'c':{'pass':False}})=='b'

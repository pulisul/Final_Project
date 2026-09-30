from pathlib import Path
from intellectagent.strict_validation import StrictCheck, summarize, sha256_file

def test_summarize_required_only(tmp_path: Path):
    checks=[StrictCheck('a',True,True),StrictCheck('b',False,False)]
    r=summarize(checks)
    assert r['overall_pass'] is True
    assert r['required_passed']==1

def test_sha256_changes(tmp_path: Path):
    p=tmp_path/'x'; p.write_text('a')
    h1=sha256_file(p); p.write_text('b'); h2=sha256_file(p)
    assert h1 != h2

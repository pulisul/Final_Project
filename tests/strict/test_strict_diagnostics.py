from pathlib import Path
from intellectagent.strict_diagnostics import disk_snapshot, classify_stage_failure


def test_disk_snapshot_has_contract(tmp_path: Path):
    r=disk_snapshot(tmp_path,min_free_gb=0.0)
    assert r['pass'] is True
    assert 'free_gb' in r


def test_classify_missing_dependency():
    r=classify_stage_failure('ragas',{'stderr':"ModuleNotFoundError: No module named 'ragas'",'stdout':''})
    assert 'missing_python_dependency' in r['categories']


def test_classify_service_unreachable():
    r=classify_stage_failure('ollama',{'stderr':'Connection refused','stdout':''})
    assert 'service_unreachable' in r['categories']

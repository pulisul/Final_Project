from pathlib import Path
from intellectagent.release_candidate import build_freeze_manifest, verify_freeze_manifest

def test_freeze_detects_change(tmp_path: Path):
    f=tmp_path/'a.txt'; f.write_text('a',encoding='utf-8')
    m=build_freeze_manifest(tmp_path,[f])
    assert verify_freeze_manifest(tmp_path,m)['pass']
    f.write_text('b',encoding='utf-8')
    r=verify_freeze_manifest(tmp_path,m)
    assert not r['pass'] and 'a.txt' in r['changed']

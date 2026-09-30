from pathlib import Path
from intellectagent.release_week9 import build_release_manifest, verify_release_manifest

def test_manifest_detects_change(tmp_path:Path):
    (tmp_path/'a.txt').write_text('a',encoding='utf-8')
    m=build_release_manifest(tmp_path,'x')
    assert verify_release_manifest(tmp_path,m)['pass']
    (tmp_path/'a.txt').write_text('b',encoding='utf-8')
    r=verify_release_manifest(tmp_path,m)
    assert not r['pass'] and 'a.txt' in r['changed']

def test_manifest_detects_missing(tmp_path:Path):
    p=tmp_path/'a.txt'; p.write_text('a',encoding='utf-8')
    m=build_release_manifest(tmp_path,'x'); p.unlink()
    r=verify_release_manifest(tmp_path,m)
    assert not r['pass'] and 'a.txt' in r['missing']

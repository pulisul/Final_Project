from pathlib import Path
from intellectagent.final_release import compare_snapshots, release_identity, snapshot_paths, atomic_write_json, active_policy_summary


def test_snapshot_detects_change(tmp_path: Path):
    (tmp_path/'config').mkdir(); p=tmp_path/'config/core.json'; p.write_text('{}')
    before=snapshot_paths(tmp_path,[p]); p.write_text('{"x":1}')
    after=snapshot_paths(tmp_path,[p]); result=compare_snapshots(before,after)
    assert result['pass'] is False
    assert 'config/core.json' in result['changed']


def test_release_identity_deterministic(tmp_path: Path):
    a=tmp_path/'a.json'; b=tmp_path/'b.json'; a.write_text('a'); b.write_text('b')
    r1=release_identity(tmp_path,[a,b]); r2=release_identity(tmp_path,[b,a])
    assert r1['release_id']==r2['release_id']


def test_atomic_json_and_policy(tmp_path: Path):
    p=tmp_path/'artifacts/week7/active_policy.json'
    atomic_write_json(p,{'active_policy':'rule','rollback_applied':True,'reason':'test'})
    s=active_policy_summary(tmp_path)
    assert s['pass'] is True and s['active_policy']=='rule'
    assert not p.with_suffix('.json.tmp').exists()

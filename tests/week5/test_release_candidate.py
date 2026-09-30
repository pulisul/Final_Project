from intellectagent.release_candidate import Severity, Issue, classify_issue, release_blockers

def test_security_issue_is_blocker():
    assert classify_issue('path escape','loader',security=True) is Severity.BLOCKER

def test_no_workaround_is_critical():
    assert classify_issue('reranker crash','retrieval',no_workaround=True) is Severity.CRITICAL

def test_minor_not_release_blocker():
    i=Issue('x','typo',Severity.MINOR,'ui')
    assert not i.blocks_release
    assert release_blockers([i]) == []

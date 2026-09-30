import socket
import pytest
from intellectagent.network_guard import LocalOnlyNetworkGuard

def test_external_network_is_blocked_before_connect():
    with LocalOnlyNetworkGuard() as g:
        with pytest.raises(RuntimeError):
            socket.create_connection(('example.com',443),timeout=.01)
    assert g.evidence()['external_attempt_count'] >= 1

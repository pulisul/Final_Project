from intellectagent.platform_runtime import RuntimeProfile, detect_runtime_profile


def test_explicit_cpu_profile_is_valid():
    p=detect_runtime_profile('cpu')
    assert p.pass_runtime is True
    assert p.torch_device=='cpu'


def test_mac_profile_rejects_non_macos(monkeypatch):
    import intellectagent.platform_runtime as pr
    monkeypatch.setattr(pr.platform,'system',lambda:'Linux')
    monkeypatch.setattr(pr.platform,'machine',lambda:'x86_64')
    p=pr.detect_runtime_profile('mac-native')
    assert p.pass_runtime is False
    assert p.reason=='mac_native_requires_macos'


def test_auto_profile_returns_supported_name():
    p=detect_runtime_profile('auto')
    assert p.profile in {'mac-native','linux-cuda','cpu'}

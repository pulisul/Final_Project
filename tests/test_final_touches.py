"""마감 전 소규모 보강: 한글 NFC 정규화, 평가용 생성 고정(temperature/seed), 앱의 오픈소스 고지."""
from __future__ import annotations

import unicodedata
from pathlib import Path

from intellectagent.generation import OllamaGenerator
from intellectagent.loaders import SafeDocumentLoader

NFC = "연차는 입사 1년 후부터 15일이 부여됩니다."
NFD = unicodedata.normalize("NFD", NFC)


def test_loader_returns_nfc_even_if_file_text_is_nfd(tmp_path):
    assert NFC != NFD and len(NFD) > len(NFC)
    for name in ("a.txt", "a.md"):
        (tmp_path / name).write_bytes(NFD.encode("utf-8"))
    ld = SafeDocumentLoader(tmp_path, (".txt", ".md"), 1_000_000)
    for name in ("a.txt", "a.md"):
        assert ld.load(tmp_path / name).text == NFC


def test_loader_keeps_nfc_text_unchanged_and_hash_is_of_the_original_bytes(tmp_path):
    from intellectagent.hashing import sha256_bytes
    (tmp_path / "a.txt").write_bytes(NFD.encode("utf-8"))
    doc = SafeDocumentLoader(tmp_path, (".txt",), 1_000_000).load(tmp_path / "a.txt")
    assert doc.sha256 == sha256_bytes(NFD.encode("utf-8"))      # 해시는 원본 바이트 기준 (정규화는 텍스트에만)


def test_pipeline_normalizes_nfd_question():
    from intellectagent.pipeline import CoreRAG
    seen = {}

    class R:
        def search(self, q, k, candidate_k=None):
            seen["q"] = q
            return []

    CoreRAG(R(), None).ask(NFD)
    assert seen["q"] == NFC


class _FakeClient:
    def __init__(self): self.calls = []
    def chat(self, **kw):
        self.calls.append(kw)
        return {"message": {"content": "x"}}


def _hit():
    from intellectagent.models import SearchHit
    return SearchHit("c1", "d1", "본문", 0.5)


def test_generator_passes_options_only_when_set():
    hit = _hit()
    c1 = _FakeClient(); OllamaGenerator("m", client=c1).generate("q", [hit])
    assert "options" not in c1.calls[0]                      # 기본 동작은 그대로
    c2 = _FakeClient(); OllamaGenerator("m", client=c2, options={"temperature": 0, "seed": 42}).generate("q", [hit])
    assert c2.calls[0]["options"] == {"temperature": 0, "seed": 42}


def test_factory_reads_generation_options_from_feature_flags():
    from intellectagent.config import CoreConfig
    from intellectagent.factory import build_components
    import inspect
    src = inspect.getsource(build_components)
    assert "ollama_temperature" in src and "ollama_seed" in src


def test_real_env_eval_cli_defaults_to_reproducible_generation():
    text = (Path(__file__).resolve().parents[1] / "scripts/autorag/real_env_eval.py").read_text(encoding="utf-8")
    assert '"--temperature"' in text and "default=0.0" in text and '"--gen-seed"' in text

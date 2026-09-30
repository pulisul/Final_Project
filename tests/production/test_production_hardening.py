from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace

from intellectagent.config import CoreConfig
from intellectagent.final_release import critical_paths, release_evidence_paths
from intellectagent.pipeline import CoreRAG
from intellectagent.production_validation import validate_production_contract
from intellectagent.strict_gpu_validation import CandidateResult, candidate_quality_pass

ROOT=Path(__file__).resolve().parents[2]


def test_production_config_is_separate_and_enables_real_stack():
    cfg=CoreConfig.load(ROOT/'config/core.production.json')
    result=validate_production_contract(cfg,ROOT/'data/real_docs')
    assert result['pass'] is True
    assert cfg.allowed_root == (ROOT/'data/real_docs').resolve()
    assert cfg.index_dir != CoreConfig.load(ROOT/'config/core.json').index_dir
    assert cfg.candidate_k >= cfg.top_k


def test_smoke_config_cannot_pass_production_contract():
    cfg=CoreConfig.load(ROOT/'config/core.json')
    result=validate_production_contract(cfg,ROOT/'data/real_docs')
    assert result['pass'] is False
    assert result['missing_required_flags']
    assert result['root_matches'] is False


def test_core_rag_propagates_candidate_k():
    class Retriever:
        def __init__(self): self.args=None
        def search(self,q,top_k,candidate_k=None):
            self.args=(q,top_k,candidate_k); return []
    class Generator:
        def generate(self,q,hits): raise AssertionError('generator should not run without evidence')
    r=Retriever(); rag=CoreRAG(r,Generator(),top_k=5,candidate_k=20)
    ans=rag.ask('q')
    assert ans.fallback is True
    assert r.args == ('q',5,20)


def test_candidate_quality_gate_rejects_zero_quality():
    weak=CandidateResult(5,0.0,0.0,1,1,1,0,0)
    strong=CandidateResult(10,0.85,0.70,2,3,1,1,0)
    assert candidate_quality_pass(weak,min_hit_rate_at_k=0.8,min_mrr=0.6) is False
    assert candidate_quality_pass(strong,min_hit_rate_at_k=0.8,min_mrr=0.6) is True


def test_release_identity_inputs_include_production_config_and_e2e():
    cps={str(p.relative_to(ROOT)) for p in critical_paths(ROOT)}
    assert 'config/core.production.json' in cps
    # production E2E is part of release evidence when present; check declared source text too.
    import intellectagent.final_release as fr
    src=(ROOT/'src/intellectagent/final_release.py').read_text(encoding='utf-8')
    assert 'evidence/strict/30_production_e2e.json' in src


def test_factory_passes_runtime_device_to_real_models(monkeypatch,tmp_path):
    import intellectagent.factory as f
    cfg=SimpleNamespace(
        feature_flags={'use_sentence_transformer':True,'use_chroma':True,'use_reranker':True,'use_ollama':True},
        runtime_profile='mac-native',embedding_model='emb',reranker_model='rr',ollama_model='llm',
        index_dir=tmp_path/'idx',collection_name='c',top_k=5,candidate_k=10,fallback_min_score=.1,
        project_root=tmp_path,
    )
    seen={}
    class E:
        def __init__(self,name,device=None): seen['emb']=device
    class S:
        def __init__(self,*a,**k): pass
    class R:
        def __init__(self,name,device=None): seen['rr']=device
    class G:
        def __init__(self,*a,**k): pass
    class Ret:
        def __init__(self,*a,**k): pass
    class Reg:
        def health(self): return []
        def apply_if_ready(self,n,p): return p
    monkeypatch.setattr(f,'SentenceTransformerEmbedder',E)
    monkeypatch.setattr(f,'ChromaVectorStore',S)
    monkeypatch.setattr(f,'CrossEncoderReranker',R)
    monkeypatch.setattr(f,'OllamaGenerator',G)
    monkeypatch.setattr(f,'Retriever',Ret)
    monkeypatch.setattr(f,'build_extension_registry',lambda cfg: Reg())
    monkeypatch.setattr(f,'detect_runtime_profile',lambda requested: SimpleNamespace(pass_runtime=True,reason='ok',torch_device='mps',profile='mac-native'))
    monkeypatch.setattr(f,'configure_mac_mps_fallback',lambda profile: None)
    f.build_components(cfg,runtime_profile='mac-native')
    assert seen == {'emb':'mps','rr':'mps'}

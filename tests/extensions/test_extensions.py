from pathlib import Path
import tempfile
from intellectagent.extensions import ExtensionRegistry, GroundingVerifierExtension, SessionPreferencesExtension, LongTermMemoryExtension, LinUCBPolicy, MultiToolAgentExtension
from intellectagent.models import Answer, SearchHit

def test_registry_and_grounding():
    r=ExtensionRegistry(); r.register(GroundingVerifierExtension(True,0.1))
    hit=SearchHit("c1","file","사과는 과일입니다.",1.0,{})
    a=Answer("q","사과는 과일입니다. [c1]",("c1",),(hit,),False)
    assert r.apply_if_ready("grounding_verifier",a).fallback is False

def test_session_preferences_roundtrip(tmp_path:Path):
    s=SessionPreferencesExtension(tmp_path/"p.db",True); s.set("x",{"tone":"short"}); assert s.get("x")["tone"]=="short"; s.delete("x"); assert s.get("x")=={}

def test_memory_requires_consent(tmp_path:Path):
    m=LongTermMemoryExtension(tmp_path/"m.db",True)
    try: m.put("u","k","v",False); assert False
    except PermissionError: pass

def test_linucb_learns_preference(tmp_path:Path):
    p=LinUCBPolicy(n_features=2,n_actions=2,alpha=0.1,enabled=True,state_path=tmp_path/"p.json")
    x=[1,0]
    for _ in range(30): p.update(1,x,1.0)
    a,_=p.choose(x); assert a==1
    p.save(); q=LinUCBPolicy(n_features=2,n_actions=2,state_path=tmp_path/"p.json"); q.load(); assert q.choose(x)[0]==1

def test_multitool_allowlist():
    a=MultiToolAgentExtension({"echo":lambda x:x},True,max_steps=3); assert a.call("echo",x=3)==3
    try: a.call("nope"); assert False
    except PermissionError: pass

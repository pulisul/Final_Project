from __future__ import annotations
import json, socket, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

@dataclass
class ServiceCheck:
    name: str
    passed: bool
    details: dict[str, Any]
    error: str | None = None
    def to_dict(self):
        d=asdict(self); d['pass']=d.pop('passed'); return d

def tcp_probe(host:str, port:int, timeout:float=1.0)->dict[str,Any]:
    t=time.perf_counter()
    try:
        with socket.create_connection((host,port),timeout=timeout):
            return {'pass':True,'host':host,'port':port,'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    except OSError as e:
        return {'pass':False,'host':host,'port':port,'elapsed_ms':round((time.perf_counter()-t)*1000,3),'error':f'{type(e).__name__}: {e}'}

def validate_grounded_text(text:str, allowed_ids:list[str])->dict[str,Any]:
    ids=[x for x in allowed_ids if f'[{x}]' in text]
    fallback='관련 근거 문서를 찾지 못했습니다' in text
    return {'pass': bool(ids) or fallback, 'citation_ids_found':ids, 'fallback':fallback, 'answer_chars':len(text)}

def load_json(path:Path)->dict[str,Any]:
    return json.loads(path.read_text(encoding='utf-8'))

def first_failed_stage(results:dict[str,dict[str,Any]])->str|None:
    for name,res in results.items():
        if res.get('pass') is not True:
            return name
    return None

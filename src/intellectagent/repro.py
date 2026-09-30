from __future__ import annotations
import json, os, platform, subprocess, sys, time
from pathlib import Path
from .hashing import sha256_file, stable_json_hash

def git_commit(root:Path)->str|None:
    try:return subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:return None

def build_manifest(root:Path,config_path:Path)->dict:
    cfg=json.loads(config_path.read_text(encoding="utf-8"))
    files={}
    for name in ["requirements.txt","requirements-lock.txt","requirements-dev.txt","pyproject.toml"]:
        p=root/name
        if p.exists(): files[name]=sha256_file(p)
    return {"created_at":time.time(),"python":sys.version,"python_executable":sys.executable,"platform":platform.platform(),
            "config_hash":stable_json_hash(cfg),"files":files,"git_commit":git_commit(root),
            "env":{"PYTHONHASHSEED":os.getenv("PYTHONHASHSEED"),"OLLAMA_HOST":os.getenv("OLLAMA_HOST")}}

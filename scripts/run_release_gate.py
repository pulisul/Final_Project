from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("PYTHONPATH",str(ROOT/"src")); sys.path.insert(0,str(ROOT/"src"))
from intellectagent.config import CoreConfig
from intellectagent.factory import build_components
from intellectagent.loaders import SafeDocumentLoader
from intellectagent.metadata import MetadataDB
from intellectagent.indexing import Indexer
from intellectagent.evaluation import load_cases,evaluate_core
from intellectagent.repro import build_manifest


def run(mode:str, config_path:Path|None=None, eval_path:Path|None=None, runtime_profile:str|None=None)->dict:
    config_path=(config_path or ROOT/"config/core.json").resolve()
    cfg=CoreConfig.load(config_path)
    policy=json.loads((ROOT/"config/release_policy.json").read_text(encoding="utf-8"))
    evidence=cfg.evidence_dir; evidence.mkdir(parents=True,exist_ok=True)
    result={"mode":mode,"config":str(config_path.relative_to(ROOT)),"overall_pass":False,"checks":{}}
    py_mm=f"{sys.version_info.major}.{sys.version_info.minor}"
    py_ok=(py_mm==policy["python_major_minor"]) if mode=="strict" else True
    result["checks"]["python_version"]={"pass":py_ok,"actual":py_mm,"required":policy["python_major_minor"],"enforced":mode=="strict"}
    p=subprocess.run([sys.executable,"-m","pytest","-q"],cwd=ROOT,text=True,capture_output=True)
    result["checks"]["pytest"]={"pass":p.returncode==0,"stdout":p.stdout[-5000:],"stderr":p.stderr[-3000:]}
    emb,store,retr,gen,pipe=build_components(cfg,runtime_profile=runtime_profile)
    loader=SafeDocumentLoader(cfg.allowed_root,cfg.allowed_extensions,cfg.max_file_bytes); db=MetadataDB(cfg.metadata_db)
    idx=Indexer(loader,emb,store,db,cfg).index_all(evidence/f"index_checkpoint_{mode}.json")
    idx_ok=idx.get("success_ratio",0)>=policy["min_index_success_ratio"]
    result["checks"]["index"]={"pass":idx_ok,"details":idx,"allowed_root":str(cfg.allowed_root)}
    if hasattr(store,"chunk_ids"):
        store_ids=set(store.chunk_ids())
    elif hasattr(store,"rows"):
        store_ids=set(getattr(store,"rows",{}).keys())
    else:
        store_ids=None
    db_ids=db.chunk_ids(); consistency=True if store_ids is None else (store_ids==db_ids)
    result["checks"]["id_consistency"]={"pass":consistency,"vector_count":None if store_ids is None else len(store_ids),"db_count":len(db_ids)}
    eval_file=(eval_path or ROOT/"data/eval/eval.jsonl").resolve()
    metrics=evaluate_core(pipe,load_cases(eval_file))
    (evidence/"core_metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
    baseline=evidence/"baseline_metrics.json"
    if mode=="smoke" and not baseline.exists(): baseline.write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
    t=policy[mode]
    mpass=(metrics["hit_rate_at_k"]>=t["min_hit_rate_at_k"] and metrics["fallback_accuracy"]>=t["min_fallback_accuracy"] and metrics["citation_success"]>=t["min_citation_success"])
    if "min_mrr" in t: mpass=mpass and metrics["mrr"]>=t["min_mrr"]
    result["checks"]["core_eval"]={"pass":mpass,"metrics":metrics,"thresholds":t,"eval":str(eval_file)}
    manifest=build_manifest(ROOT,config_path)
    manifest_path=evidence/("REPRO_MANIFEST.json" if mode=="strict" else "SMOKE_REPRO_MANIFEST.json")
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    result["checks"]["repro_manifest"]={"pass":bool(manifest["config_hash"] and manifest["files"]),"manifest":manifest}
    health=[]
    if getattr(pipe,"extensions",None) is not None:
        health=[h.__dict__ for h in pipe.extensions.health()]
    result["checks"]["extension_isolation"]={"pass":all((not h["enabled"]) or h["ready"] for h in health),"health":health}
    if mode=="strict":
        lock=ROOT/"requirements-lock.txt"; lock_ok=lock.exists() and lock.stat().st_size>0
        result["checks"]["requirements_lock"]={"pass":lock_ok,"path":str(lock)}
        pc=subprocess.run([sys.executable,"-m","pip","check"],cwd=ROOT,text=True,capture_output=True)
        result["checks"]["pip_check"]={"pass":pc.returncode==0,"stdout":pc.stdout[-3000:],"stderr":pc.stderr[-3000:]}
    result["overall_pass"]=all(v["pass"] for v in result["checks"].values())
    outfile=evidence/("FINAL_GATE.json" if mode=="strict" else "SMOKE_GATE.json")
    outfile.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["smoke","strict"],default="strict")
    ap.add_argument("--config",default="config/core.json")
    ap.add_argument("--eval",default=None)
    ap.add_argument("--profile",choices=["auto","mac-native","linux-cuda","cpu"],default=None)
    a=ap.parse_args()
    r=run(a.mode,ROOT/a.config,(ROOT/a.eval if a.eval else None),a.profile)
    print(json.dumps(r,ensure_ascii=False,indent=2)); return 0 if r["overall_pass"] else 2
if __name__=="__main__": raise SystemExit(main())

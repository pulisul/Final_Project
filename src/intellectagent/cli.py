from __future__ import annotations
import argparse,json
from pathlib import Path
from .config import CoreConfig
from .factory import build_components
from .loaders import SafeDocumentLoader
from .metadata import MetadataDB
from .indexing import Indexer
from .evaluation import load_cases,evaluate_core
from .repro import build_manifest

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config/core.json")
    sub=ap.add_subparsers(dest="cmd",required=True)
    sub.add_parser("index")
    ask=sub.add_parser("ask"); ask.add_argument("question")
    ev=sub.add_parser("evaluate"); ev.add_argument("--cases",default="data/eval/eval.jsonl")
    sub.add_parser("manifest")
    a=ap.parse_args(); cfg=CoreConfig.load(Path(a.config)); emb,store,retr,gen,pipe=build_components(cfg)
    loader=SafeDocumentLoader(cfg.allowed_root,cfg.allowed_extensions,cfg.max_file_bytes); db=MetadataDB(cfg.metadata_db)
    # In-memory smoke mode must index in the same process before ask/evaluate.
    if not cfg.feature_flags.get("use_chroma") and a.cmd in {"ask","evaluate"}:
        Indexer(loader,emb,store,db,cfg).index_all(cfg.evidence_dir/"index_checkpoint.json")
    if a.cmd=="index": print(json.dumps(Indexer(loader,emb,store,db,cfg).index_all(cfg.evidence_dir/"index_checkpoint.json"),ensure_ascii=False,indent=2))
    elif a.cmd=="ask": print(pipe.ask(a.question))
    elif a.cmd=="evaluate": print(json.dumps(evaluate_core(pipe,load_cases(cfg.project_root/a.cases)),ensure_ascii=False,indent=2))
    elif a.cmd=="manifest":
        m=build_manifest(cfg.project_root,Path(a.config)); cfg.evidence_dir.mkdir(parents=True,exist_ok=True); (cfg.evidence_dir/"REPRO_MANIFEST.json").write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(m,ensure_ascii=False,indent=2))
if __name__=="__main__":main()

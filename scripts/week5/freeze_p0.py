#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from intellectagent.release_candidate import build_freeze_manifest, write_json

CORE_FILES = [
    'config/core.json','config/release_policy.json','requirements.txt','requirements-dev.txt',
    'src/intellectagent/loaders.py','src/intellectagent/chunking.py','src/intellectagent/embeddings.py',
    'src/intellectagent/vectorstore.py','src/intellectagent/metadata.py','src/intellectagent/indexing.py',
    'src/intellectagent/retrieval.py','src/intellectagent/reranking.py','src/intellectagent/generation.py',
    'src/intellectagent/grounding.py','src/intellectagent/orchestration.py','src/intellectagent/pipeline.py',
    'src/intellectagent/evaluation.py','scripts/run_release_gate.py'
]

def main():
    missing=[f for f in CORE_FILES if not (ROOT/f).exists()]
    if missing:
        raise SystemExit('missing core files: '+', '.join(missing))
    m=build_freeze_manifest(ROOT,[ROOT/f for f in CORE_FILES],label='P0-RC1')
    out=ROOT/'evidence/week5/P0_FREEZE.json'; write_json(out,m)
    print(out)
if __name__=='__main__': main()

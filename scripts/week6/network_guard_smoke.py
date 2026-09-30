#!/usr/bin/env python3
from __future__ import annotations
import json, sys, socket
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from intellectagent.network_guard import LocalOnlyNetworkGuard
out_path=ROOT/'evidence/week6/network_guard.json'
blocked=False
with LocalOnlyNetworkGuard() as guard:
    try:
        socket.create_connection(('example.com',443),timeout=0.1)
    except RuntimeError:
        blocked=True
    except Exception:
        # local DNS/transport behavior is not itself the assertion; the guard attempt log is.
        pass
ev=guard.evidence(); ev['synthetic_external_block_test_pass']=blocked
guard.write_evidence(out_path)
# rewrite with synthetic test field included
out_path.write_text(json.dumps(ev,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(ev,ensure_ascii=False,indent=2))
raise SystemExit(0 if blocked else 1)

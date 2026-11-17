#!/usr/bin/env python3
from __future__ import annotations
import json, html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
e=ROOT/'evidence'; out=ROOT/'artifacts/regression_dashboard.html'
files=sorted(e.glob('*metrics*.json'))
rows=[]
for f in files:
    try: d=json.loads(f.read_text(encoding='utf-8'))
    except Exception: continue
    rows.append((f.name,d))
metrics=['hit_rate_at_k','mrr','fallback_accuracy','citation_success','latency_ms_mean','latency_ms_p95','reward_mean']
parts=['<html><meta charset="utf-8"><title>IntellectAgent Regression Dashboard</title><body>','<h1>Regression Dashboard</h1>','<table border="1" cellspacing="0" cellpadding="6"><tr><th>run</th>'+''.join(f'<th>{m}</th>' for m in metrics)+'</tr>']
for name,d in rows:
    parts.append('<tr><td>'+html.escape(name)+'</td>'+''.join(f'<td>{html.escape(str(d.get(m,"")))}</td>' for m in metrics)+'</tr>')
parts.append('</table></body></html>'); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(''.join(parts),encoding='utf-8'); print(out)

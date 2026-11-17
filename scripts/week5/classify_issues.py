#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from intellectagent.release_candidate import Severity, Issue, classify_issue, issue_to_dict

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--title',required=True); ap.add_argument('--component',default='core')
    ap.add_argument('--data-loss',action='store_true'); ap.add_argument('--security',action='store_true')
    ap.add_argument('--core-unusable',action='store_true'); ap.add_argument('--no-workaround',action='store_true')
    ap.add_argument('--cosmetic',action='store_true'); ap.add_argument('--workaround')
    a=ap.parse_args()
    sev=classify_issue(a.title,a.component,data_loss=a.data_loss,security=a.security,core_unusable=a.core_unusable,no_workaround=a.no_workaround,cosmetic=a.cosmetic)
    issue=Issue('W5-'+uuid.uuid4().hex[:8],a.title,sev,a.component,workaround=a.workaround)
    out=ROOT/'evidence/week5/issues.jsonl'; out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('a',encoding='utf-8') as f:f.write(json.dumps(issue_to_dict(issue),ensure_ascii=False)+'\n')
    print(json.dumps(issue_to_dict(issue),ensure_ascii=False,indent=2))
if __name__=='__main__': main()

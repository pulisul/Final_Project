from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; E=ROOT/'evidence/week9'; E.mkdir(parents=True,exist_ok=True)

def run(cmd):
    p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
    return {'pass':p.returncode==0,'returncode':p.returncode,'stdout':p.stdout[-5000:],'stderr':p.stderr[-2000:]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--strict',action='store_true'); a=ap.parse_args()
    checks={}
    checks['final_regression']=run([sys.executable,'scripts/week9/run_final_regression.py'])
    checks['submission_manifest']=run([sys.executable,'scripts/week9/build_submission_manifest.py'])
    checks['release_manifest']=run([sys.executable,'scripts/week9/build_release_manifest.py'])
    # Preserve prior smoke gates as mandatory evidence.
    for w in range(2,9):
        p=ROOT/f'evidence/week{w}/WEEK{w}_GATE.json'
        ok=False
        if p.exists():
            try: ok=bool(json.loads(p.read_text(encoding='utf-8')).get('overall_pass',False))
            except Exception: ok=False
        checks[f'week{w}_gate']={'pass':ok,'path':str(p.relative_to(ROOT))}
    if a.strict:
        checks['clean_repro']=run([sys.executable,'scripts/week9/check_clean_repro.py'])
        checks['core_strict']=run([sys.executable,'scripts/run_release_gate.py','--mode','strict'])
        checks['week4_strict']=run([sys.executable,'scripts/week4/run_week4_gate.py','--strict'])
        checks['week6_strict']=run([sys.executable,'scripts/week6/run_week6_gate.py','--strict-ragas'])
        checks['week8_strict']=run([sys.executable,'scripts/week8/run_week8_gate.py','--strict-streamlit'])
    overall=all(v['pass'] for v in checks.values())
    r={'mode':'strict' if a.strict else 'smoke','overall_pass':overall,'checks':checks}
    out=E/('WEEK9_STRICT_GATE.json' if a.strict else 'WEEK9_GATE.json')
    out.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(r,ensure_ascii=False,indent=2)); return 0 if overall else 2
if __name__=='__main__': raise SystemExit(main())

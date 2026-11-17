#!/usr/bin/env python3
from __future__ import annotations
import json,subprocess,sys,time,socket
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def port_open(port):
    try:
        with socket.create_connection(('127.0.0.1',port),timeout=.5): return True
    except OSError:return False

def main():
    ev={'stage':'streamlit_runtime','pass':False}; port=18501
    try:
        import streamlit
        ev['streamlit_version']=streamlit.__version__
        p=subprocess.Popen([sys.executable,'-m','streamlit','run','app_streamlit.py','--server.headless=true',f'--server.port={port}','--browser.gatherUsageStats=false'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        ok=False
        for _ in range(30):
            if p.poll() is not None: break
            if port_open(port): ok=True; break
            time.sleep(.5)
        ev['port']=port; ev['port_open']=ok; ev['process_alive']=p.poll() is None
        ev['pass']=ok
        p.terminate()
        try:p.wait(timeout=5)
        except subprocess.TimeoutExpired:p.kill()
        if not ok:
            out,err=p.communicate(timeout=2); ev['stdout']=out[-3000:]; ev['stderr']=err[-3000:]
    except Exception as e: ev['error']=f'{type(e).__name__}: {e}'
    out=ROOT/'evidence/strict/24_streamlit_runtime.json'; out.write_text(json.dumps(ev,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(ev,ensure_ascii=False,indent=2)); return 0 if ev['pass'] else 2
if __name__=='__main__': raise SystemExit(main())

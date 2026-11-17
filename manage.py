#!/usr/bin/env python3
"""IntellectAgent 실행기 — 설치 뒤 모든 작업의 입구 (macOS · Linux · Windows 공통).

  bash run.sh <명령>        (macOS/Linux)        run.bat <명령>      (Windows)

명령
  doctor              환경 진단(파이썬·패키지·Ollama·모델·디스크)
  app                 화면 실행 → http://localhost:8501
  test [pytest 인자]  테스트
  redteam             반대 입장(레드팀) 시뮬레이션 53개 시나리오
  docs [문서폴더]     내 문서가 제대로 읽히는지 점검 (모델 불필요)
  qa   [문서폴더]     질문셋 초안 생성 (Ollama 필요)
  eval [문서폴더]     검색·거절·종단 평가 (승인된 질문셋 필요). 환경변수: E2E CHUNKS RETRIEVERS CITATION_STYLE EVAL_OUT
  setup-model         Ollama 서버 확인 + 답변 모델 내려받기
이 파일은 표준 라이브러리만 쓰므로 어떤 파이썬 3.8+ 로도 시작할 수 있다. 실제 작업은 .venv-real 안의 파이썬이 한다.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import envtools as E  # noqa: E402

ROOT = E.ROOT
MODEL = os.environ.get("OLLAMA_MODEL", E.DEFAULT_MODEL)


def die(msg: str, code: int = 2):
    print(f"[FAIL] {msg}")
    sys.exit(code)


def vpy() -> Path:
    py = E.venv_python()
    if not py.exists():
        die("가상환경이 없습니다. 먼저 설치하세요:  bash install.sh   (Windows: install.bat)")
    return py


def venv_run(args: list, extra_env: dict | None = None, check: bool = True) -> int:
    env = E.runtime_env(); env.update(extra_env or {})
    return E.run([vpy()] + [str(a) for a in args], env=env, check=check)


def ensure_ollama() -> None:
    if E.ollama_up():
        return
    if not E.ollama_installed():
        die(f"Ollama 가 없습니다: {E.ollama_install_hint()}")
    print("Ollama 서버를 시작합니다...")
    if not E.start_ollama():
        die("Ollama 서버가 시작되지 않았습니다. 터미널에서 `ollama serve` 를 직접 실행해 보세요.")


def docs_dir(arg: str | None) -> Path:
    d = Path(arg or os.environ.get("DOCS") or ROOT / "data" / "real_docs").expanduser().resolve()
    if not d.is_dir() or not any(p for p in d.rglob("*") if p.is_file() and not p.name.startswith(".")):
        die(f"문서 폴더가 없거나 비었습니다: {d}\n  → 문서를 이 폴더에 넣거나, 폴더 경로를 명령 뒤에 적으세요 (예: run.sh docs ~/Documents/업무문서)")
    return d


def cmd_app(a):
    ensure_ollama() if not a.no_ollama else None
    cfg = os.environ.get("INTELLECTAGENT_CONFIG") or str(ROOT / "config" / "core.production.json")
    if not Path(cfg).exists():
        die(f"설정 파일이 없습니다: {cfg}")
    print(f"→ http://localhost:{a.port}  (종료: Ctrl+C)")
    rc = venv_run(["-m", "streamlit", "run", "app_streamlit.py", "--server.port", a.port, "--browser.gatherUsageStats", "false"],
                  {"INTELLECTAGENT_CONFIG": cfg}, check=False)
    sys.exit(rc)


def cmd_test(a):
    sys.exit(venv_run(["-m", "pytest", "-p", "no:cacheprovider"] + a.rest, check=False))


def cmd_redteam(a):
    out = ROOT / "evidence" / "redteam"; out.mkdir(parents=True, exist_ok=True)
    rc = venv_run(["-I", "scripts/redteam/redteam.py", ROOT, out], check=False)
    try:
        res = json.loads((out / "results.json").read_text(encoding="utf-8"))
        open_ = [r for r in res if r["verdict"] in ("BREACHED", "PARTIAL", "UNTESTABLE")]
        print(f"\n남은 항목 {len(open_)}개 (자세한 설명: docs/REDTEAM_REPORT.md)")
    except Exception:
        pass
    sys.exit(rc)


def cmd_docs(a):
    d = docs_dir(a.folder); (ROOT / "evidence").mkdir(exist_ok=True)
    print("※ 문서마다 <파일명>.expect.txt(본문 문장 3~5줄)를 두면 OK 판정이 나옵니다. 없으면 OK_UNVERIFIED.")
    rc = venv_run(["scripts/autorag/check_real_docs.py", d, "--out", "evidence/real_docs_check", "--dump-dir", "evidence/real_docs_dump"], check=False)
    print("→ evidence/real_docs_check/ 보고서와 evidence/real_docs_dump/*.extracted.txt 를 눈으로 확인하세요")
    sys.exit(rc)


def cmd_qa(a):
    d = docs_dir(a.folder); ensure_ollama()
    cache = Path(os.environ.get("PARSE_CACHE") or Path.home() / ".intellectagent_parse_cache"); cache.mkdir(parents=True, exist_ok=True)
    rc = venv_run(["scripts/autorag/gen_qa.py", "--root", d, "--out", "evidence/qa", "--model", MODEL, "--n", "60", "--unanswerable", "10",
                   "--oracle", "--parse-cache", cache], check=False)
    print('→ evidence/qa/qa_draft.jsonl 을 읽고, 좋은 줄만 "status": "approved" 로 바꾼 뒤 eval 을 실행하세요 (scripts/review_qa.py 로 대화식 승인 가능)')
    sys.exit(rc)


def cmd_eval(a):
    d = docs_dir(a.folder); ensure_ollama()
    qa = ROOT / "evidence" / "qa" / "qa_draft.jsonl"
    if not qa.exists() or not re.search(r'"status":\s*"approved"', qa.read_text(encoding="utf-8")):
        die("승인(approved)된 질문이 없습니다. 먼저 qa 를 실행하고 질문을 승인하세요.")
    cache = Path(os.environ.get("PARSE_CACHE") or Path.home() / ".intellectagent_parse_cache"); cache.mkdir(parents=True, exist_ok=True)
    out = os.environ.get("EVAL_OUT", "evidence/real_eval")
    rc = venv_run(["scripts/autorag/real_env_eval.py", "--root", d, "--qa", qa, "--out", out,
                   "--chunks", os.environ.get("CHUNKS", "500:50,700:80"), "--retrievers", os.environ.get("RETRIEVERS", "vector,hybrid_rrf,bge_m3"),
                   "--top-k", "5", "--e2e", os.environ.get("E2E", "20"),
                   "--citation-style", os.environ.get("CITATION_STYLE", "index")], check=False)
    print(f"종료 코드 {rc}  (0=PASS 1=FAIL 2=입력/환경 오류 3=INVALID)  → {out}/report.md")
    sys.exit(rc)


def cmd_setup_model(a):
    ensure_ollama()
    if E.has_model(MODEL):
        print(f"모델 {MODEL} 이미 있음"); return
    E.run([E.ollama_installed() or "ollama", "pull", MODEL])


def cmd_doctor(a):
    rows: list[tuple[str, str, str]] = []
    add = lambda lv, name, note="": rows.append((lv, name, note))   # noqa: E731
    info = E.system_info()
    add("OK", "운영체제", f"{info['os']} {info['release']} / {info['arch']}")
    add("OK" if (info["ram_gb"] or 0) >= 16 else "WARN", "메모리", f"{info['ram_gb']}GB (14B 모델은 16GB 이상 권장)")
    add("OK" if info["free_disk_gb"] >= 10 else "WARN", "여유 디스크", f"{info['free_disk_gb']}GB")
    py = E.venv_python()
    if not py.exists():
        add("FAIL", "가상환경", f"없음 → 설치: bash install.sh / install.bat")
    else:
        add("OK" if E.is_py311(str(py)) else "FAIL", "가상환경 파이썬", str(py))
        code = ("import json,importlib\nr={}\n"
                "for m in ['streamlit','chromadb','FlagEmbedding','olefile','langgraph','ragas','sentence_transformers','pypdf']:\n"
                "    try: importlib.import_module(m); r[m]='ok'\n    except Exception as e: r[m]=type(e).__name__\n"
                "try:\n    import torch\n    r['torch']=torch.__version__+(' mps' if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available() else ' cuda' if torch.cuda.is_available() else ' cpu')\n"
                "except Exception as e: r['torch']=type(e).__name__\nprint(json.dumps(r))\n")
        import subprocess
        try:
            out = subprocess.check_output([str(py), "-c", code], text=True, env={**os.environ, **E.runtime_env()}, timeout=180).strip().splitlines()[-1]
            for k, v in json.loads(out).items():
                add("OK" if v == "ok" or k == "torch" and "Error" not in v else "FAIL", f"패키지 {k}", v)
        except Exception as ex:  # noqa: BLE001
            add("FAIL", "패키지 점검", f"{type(ex).__name__}: {ex}")
    add("OK" if E.ollama_installed() else "WARN", "Ollama 설치", E.ollama_installed() or E.ollama_install_hint())
    up = E.ollama_up()
    add("OK" if up else "WARN", "Ollama 서버", "응답함" if up else "꺼져 있음(app/qa/eval 실행 때 자동 시작)")
    if up:
        add("OK" if E.has_model(MODEL) else "WARN", f"모델 {MODEL}", "있음" if E.has_model(MODEL) else "없음 → run.sh setup-model")
    hf = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub" / "models--BAAI--bge-m3"
    add("OK" if hf.exists() else "WARN", "BGE-M3 임베딩 모델", "내려받음" if hf.exists() else "없음(처음 사용할 때 자동으로 내려받음)")
    cfg = ROOT / "config" / "core.production.json"
    add("OK" if cfg.exists() else "FAIL", "설정 파일", str(cfg.relative_to(ROOT)) if cfg.exists() else "없음")
    n = sum(1 for p in (ROOT / "data" / "real_docs").rglob("*") if p.is_file() and not p.name.startswith(".")) if (ROOT / "data" / "real_docs").is_dir() else 0
    add("OK" if n else "WARN", "내 문서(data/real_docs)", f"{n}개" if n else "비어 있음(화면에서 폴더를 직접 지정할 수도 있음)")
    w = max(len(r[1]) for r in rows)
    for lv, name, note in rows:
        print(f"[{lv:4}] {name:<{w}}  {note}")
    sys.exit(1 if any(r[0] == "FAIL" for r in rows) else 0)


def main() -> None:
    E.utf8_console()
    ap = argparse.ArgumentParser(description="IntellectAgent 실행기", epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("app"); p.add_argument("--port", default=os.environ.get("PORT", "8501")); p.add_argument("--no-ollama", action="store_true"); p.set_defaults(f=cmd_app)
    p = sub.add_parser("test"); p.add_argument("rest", nargs=argparse.REMAINDER); p.set_defaults(f=cmd_test)
    sub.add_parser("redteam").set_defaults(f=cmd_redteam)
    for n, f in (("docs", cmd_docs), ("qa", cmd_qa), ("eval", cmd_eval)):
        p = sub.add_parser(n); p.add_argument("folder", nargs="?"); p.set_defaults(f=f)
    sub.add_parser("setup-model").set_defaults(f=cmd_setup_model)
    sub.add_parser("doctor").set_defaults(f=cmd_doctor)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()

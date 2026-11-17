#!/usr/bin/env python3
"""IntellectAgent 설치기 — 압축을 푼 뒤 한 번만 실행한다 (macOS · Linux · Windows).

  macOS/Linux :  bash install.sh
  Windows     :  install.bat

하는 일 (순서대로)
  1. 컴퓨터 점검      OS·CPU·메모리·디스크·GPU, 파이썬 3.11 여부
  2. 가상환경         .venv-real 만들기
  3. torch            OS 에 맞는 것 (Apple 실리콘=MPS, NVIDIA=CUDA, 그 외 CPU)
  4. 패키지           requirements-dev.txt + FlagEmbedding(BGE-M3)
  5. Ollama           설치 확인(동의하면 설치) → 서버 시작 → 답변 모델 내려받기
  6. 임베딩 모델       BGE-M3 미리 내려받기
  7. 시험             테스트 + 레드팀(반대 입장) 시뮬레이션
  → 요약과 다음 명령을 보여 주고 install_report.json / install.log 를 남긴다.

옵션
  --yes               묻지 않고 진행 (Ollama 같은 시스템 프로그램 설치는 --install-ollama 도 함께 줘야 함)
  --dry-run           아무것도 바꾸지 않고 할 일만 출력
  --skip-model        Ollama 모델 내려받기 생략       --model NAME   기본 qwen2.5:14b
  --skip-embedding    BGE-M3 내려받기 생략            --with-reranker  재순위 모델도 내려받기
  --skip-tests        시험 생략                       --recreate     가상환경을 지우고 다시 만들기
  --install-ollama    Ollama 가 없으면 설치 시도       --venv PATH    가상환경 위치
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import envtools as E  # noqa: E402

ROOT = E.ROOT
LOG = ROOT / "install.log"
RESULTS: list[dict] = []
ARGS: argparse.Namespace


def say(msg: str = "") -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def sh(cmd: list, *, env: dict | None = None, check: bool = True, cwd: Path | None = None) -> int:
    """명령을 실행하고 출력을 화면과 install.log 에 함께 남긴다. --dry-run 이면 출력만 한다."""
    text = "  $ " + " ".join(str(c) for c in cmd)
    say(text)
    if ARGS.dry_run:
        return 0
    e = os.environ.copy(); e.update(E.runtime_env()); e.update(env or {})
    p = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd or ROOT), env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    with LOG.open("a", encoding="utf-8") as f:
        for line in p.stdout:                                  # type: ignore[union-attr]
            print("    " + line.rstrip(), flush=True); f.write(line)
    rc = p.wait()
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return rc


def ask(question: str, default: bool = False) -> bool:
    if ARGS.yes:
        return True
    if not sys.stdin.isatty():
        return default
    try:
        a = input(f"{question} [{'Y/n' if default else 'y/N'}] ").strip().lower()
    except EOFError:
        return default
    return default if not a else a in ("y", "yes", "예", "네", "ㅇ")


def step(name: str, critical: bool = True):
    """단계 실행 결과를 기록하는 데코레이터. critical 단계가 실패하면 설치를 멈춘다."""
    def deco(fn):
        def wrapper(*a, **k):
            say(f"\n━━ {name} ━━")
            t = time.time()
            try:
                note = fn(*a, **k) or ""
                status = "SKIP" if str(note).startswith("건너뜀") else "OK"
                RESULTS.append({"step": name, "status": status, "note": str(note), "sec": round(time.time() - t, 1)})
                say(f"[{status}] {name} {note}")
                return True
            except Exception as ex:  # noqa: BLE001
                RESULTS.append({"step": name, "status": "FAIL", "note": f"{type(ex).__name__}: {ex}", "sec": round(time.time() - t, 1)})
                say(f"[FAIL] {name}: {type(ex).__name__}: {ex}")
                if critical:
                    finish(1)
                return False
        return wrapper
    return deco


# ------------------------------------------------------------------ 1. 점검
@step("1/7 컴퓨터 점검")
def check_system():
    info = E.system_info()
    say("  " + json.dumps(info, ensure_ascii=False))
    if sys.version_info[:2] != E.PY_REQUIRED:
        raise RuntimeError(f"파이썬 3.11 이 필요합니다(지금 {sys.version.split()[0]}). install.sh / install.bat 로 실행하면 3.11 을 찾아 씁니다.")
    warn = []
    if info["os"] not in ("macos", "linux", "windows"):
        warn.append(f"검증하지 않은 OS: {info['os']}")
    if info["ram_gb"] and info["ram_gb"] < 16 and not ARGS.skip_model and ARGS.model == E.DEFAULT_MODEL:
        warn.append(f"메모리 {info['ram_gb']}GB: 14B 모델은 16GB 이상 권장 → 느리거나 실패할 수 있음 (--model qwen2.5:7b 로 가벼운 모델 선택 가능)")
    if info["free_disk_gb"] < 25:
        warn.append(f"여유 디스크 {info['free_disk_gb']}GB: 모델·패키지에 약 20GB 필요")
    for w in warn:
        say("  [주의] " + w)
    if warn and not ARGS.yes and sys.stdin.isatty() and not ask("그래도 계속할까요?", True):
        raise RuntimeError("사용자가 중단")
    return f"{info['os']} {info['arch']}, RAM {info['ram_gb']}GB, 디스크 여유 {info['free_disk_gb']}GB"


# ------------------------------------------------------------------ 2. 가상환경
@step("2/7 가상환경")
def make_venv():
    v = E.venv_dir(ARGS.venv)
    if ARGS.recreate and v.exists():
        say(f"  기존 가상환경 삭제: {v}")
        if not ARGS.dry_run:
            shutil.rmtree(v)
    if not E.venv_python(v).exists():
        try:
            sh([sys.executable, "-m", "venv", v])
        except subprocess.CalledProcessError:
            hint = {"linux": "sudo apt install python3.11-venv   (우분투/데비안 계열)", "macos": "brew reinstall python@3.11",
                    "windows": "python.org 의 3.11 설치 관리자로 다시 설치(Add to PATH 체크)"}.get(E.os_kind(), "")
            raise RuntimeError(f"가상환경을 만들지 못했습니다. {hint}")
    py = E.venv_python(v)
    if not ARGS.dry_run and not E.is_py311(str(py)):
        raise RuntimeError(f"가상환경 파이썬이 3.11 이 아닙니다: {py}  (--recreate 로 다시 만드세요)")
    sh([py, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    return str(v)


def vpy() -> Path:
    return E.venv_python(E.venv_dir(ARGS.venv))


# ------------------------------------------------------------------ 3. torch
@step("3/7 torch")
def install_torch():
    info = E.system_info(); py = vpy()
    base = [py, "-m", "pip", "install", "torch"]
    if info["os"] == "linux" and not info["nvidia_gpu"] and info["arch"] in ("x86_64", "AMD64"):
        say("  NVIDIA GPU 없음 → CPU 전용 torch(작음)를 먼저 시도")
        if sh(base + ["--index-url", "https://download.pytorch.org/whl/cpu"], check=False) != 0:
            say("  CPU 전용 저장소 접근 실패 → 일반 torch 로 대체(더 크지만 동작은 같음)")
            sh(base)
    else:
        sh(base)
    if ARGS.dry_run:
        return "(dry-run)"
    # 실제로 설치된 것을 확인해서 보고한다 (이미 설치돼 있으면 pip 가 아무것도 안 하므로 추측하지 않는다)
    out = subprocess.run([str(py), "-c", "import torch;print(torch.__version__);print(int(bool(getattr(torch.backends,'mps',None) and torch.backends.mps.is_available())));print(int(torch.cuda.is_available()))"],
                         capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        raise RuntimeError("torch 를 불러오지 못했습니다: " + out.stderr.strip()[-200:])
    ver, mps, cuda = (out.stdout.split() + ["?", "0", "0"])[:3]
    return f"{ver} ({'MPS' if mps == '1' else 'CUDA' if cuda == '1' else 'CPU'} 사용 가능)"


# ------------------------------------------------------------------ 4. 패키지
@step("4/7 패키지")
def install_packages():
    py = vpy()
    sh([py, "-m", "pip", "install", "-r", "requirements-dev.txt"])
    sh([py, "-m", "pip", "install", "FlagEmbedding"])
    rc = sh([py, "-m", "pip", "check"], check=False)
    return "의존성 충돌 없음" if rc == 0 else "pip check 경고가 있음(위 로그 확인)"


# ------------------------------------------------------------------ 5. Ollama
@step("5/7 Ollama와 답변 모델", critical=False)
def setup_ollama():
    if ARGS.skip_model:
        return "건너뜀(--skip-model)"
    exe = E.ollama_installed()
    if not exe:
        say(f"  Ollama 가 없습니다. 설치 방법: {E.ollama_install_hint()}")
        if not ARGS.install_ollama and not (not ARGS.yes and ask("지금 설치할까요? (시스템 프로그램 설치)", False)):
            raise RuntimeError("Ollama 없음 — 설치 후 `bash run.sh setup-model`(또는 run.bat setup-model)을 실행하세요")
        k = E.os_kind()
        if k == "macos" and shutil.which("brew"):
            sh(["brew", "install", "ollama"])
        elif k == "linux":
            sh(["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"])
        elif k == "windows" and shutil.which("winget"):
            sh(["winget", "install", "-e", "--id", "Ollama.Ollama", "--accept-package-agreements", "--accept-source-agreements"])
        else:
            raise RuntimeError(f"자동 설치 수단이 없습니다. 직접 설치하세요: {E.ollama_install_hint()}")
        exe = E.ollama_installed()
        if not exe and not ARGS.dry_run:
            raise RuntimeError("설치 후에도 ollama 를 찾지 못했습니다(새 터미널에서 다시 실행해 보세요)")
    if not ARGS.dry_run and not E.start_ollama():
        raise RuntimeError("Ollama 서버를 시작하지 못했습니다(`ollama serve` 를 직접 실행해 보세요)")
    if not ARGS.dry_run and E.has_model(ARGS.model):
        return f"모델 {ARGS.model} 이미 있음"
    say(f"  모델 {ARGS.model} 내려받기(수 GB, 시간이 걸립니다)")
    sh([exe or "ollama", "pull", ARGS.model])
    return f"모델 {ARGS.model} 준비"


# ------------------------------------------------------------------ 6. 임베딩 모델
@step("6/7 임베딩 모델(BGE-M3)", critical=False)
def prefetch_embedding():
    if ARGS.skip_embedding:
        return "건너뜀(--skip-embedding)"
    ids = ["BAAI/bge-m3"] + (["BAAI/bge-reranker-v2-m3"] if ARGS.with_reranker else [])
    code = ("import sys\nfrom huggingface_hub import snapshot_download\n"
            "for m in sys.argv[1:]:\n    print('다운로드', m, flush=True); print(snapshot_download(m))\n")
    sh([vpy(), "-c", code] + ids)
    return ", ".join(ids)


# ------------------------------------------------------------------ 7. 시험
@step("7/7 시험(테스트 + 레드팀)", critical=False)
def run_checks():
    if ARGS.skip_tests:
        return "건너뜀(--skip-tests)"
    py = vpy()
    sh([py, "-m", "pytest", "-p", "no:cacheprovider"])      # pyproject addopts 에 -q 가 이미 있다
    out = ROOT / "evidence" / "redteam"
    out.mkdir(parents=True, exist_ok=True)
    sh([py, "-I", "scripts/redteam/redteam.py", str(ROOT), str(out)])
    return f"통과 · 레드팀 결과 {out / 'results.json'}"


def finish(code: int) -> None:
    ok = [r for r in RESULTS if r["status"] == "OK"]; bad = [r for r in RESULTS if r["status"] == "FAIL"]
    say("\n" + "═" * 60)
    say("설치 요약")
    for r in RESULTS:
        say(f"  [{r['status']:4}] {r['step']}  {r['note']}")
    ex = "bash run.sh" if E.os_kind() != "windows" else "run.bat"
    say("\n다음 단계")
    if bad:
        say("  - FAIL 항목을 고친 뒤 같은 명령을 다시 실행하세요(이미 끝난 단계는 빨리 지나갑니다). 자세한 기록: install.log")
    say(f"  {ex} doctor       환경 진단")
    say(f"  {ex} app          화면 실행 (http://localhost:8501)")
    say(f"  {ex} redteam      반대 입장 시뮬레이션 다시 돌리기")
    say(f"  {ex} docs <폴더>  내 문서 읽기 점검")
    if not ARGS.dry_run:
        (ROOT / "install_report.json").write_text(json.dumps({"results": RESULTS, "system": E.system_info(),
                                                              "python": sys.version, "args": vars(ARGS)}, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.exit(code)


def main() -> None:
    global ARGS
    E.utf8_console()
    ap = argparse.ArgumentParser(description="IntellectAgent 설치기", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--yes", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-model", action="store_true"); ap.add_argument("--skip-embedding", action="store_true")
    ap.add_argument("--skip-tests", action="store_true"); ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--install-ollama", action="store_true"); ap.add_argument("--with-reranker", action="store_true")
    ap.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", E.DEFAULT_MODEL)); ap.add_argument("--venv", default=None)
    ARGS = ap.parse_args()
    LOG.write_text(f"IntellectAgent install {time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")
    say("IntellectAgent 설치를 시작합니다" + (" (dry-run: 실제로 바꾸지 않음)" if ARGS.dry_run else ""))
    check_system(); make_venv(); install_torch(); install_packages()
    setup_ollama(); prefetch_embedding(); run_checks()
    finish(1 if any(r["status"] == "FAIL" and r["step"].startswith(("1", "2", "3", "4")) for r in RESULTS) else 0)


if __name__ == "__main__":
    main()

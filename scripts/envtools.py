"""설치(setup_env.py)와 실행(manage.py)이 함께 쓰는 도구. 표준 라이브러리만 사용한다(venv 만들기 전에도 돌아야 하므로).

macOS · Linux · Windows 를 구분하는 일, 가상환경 파이썬 경로, Ollama 서버 확인·기동은 여기 한 곳에만 둔다.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_NAME = ".venv-real"                     # 기존 run_real_env.sh · README 와 같은 이름
OLLAMA_URL = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = "qwen2.5:14b"
PY_REQUIRED = (3, 11)                         # pyproject: >=3.11,<3.12


def utf8_console() -> None:
    """윈도우 콘솔에서도 한글이 깨지지 않게 한다."""
    os.environ.setdefault("PYTHONUTF8", "1")
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def os_kind() -> str:
    return {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}.get(platform.system(), platform.system().lower())


def system_info() -> dict:
    info = {"os": os_kind(), "release": platform.release(), "arch": platform.machine(),
            "python": platform.python_version(), "cpu_count": os.cpu_count() or 0}
    info["ram_gb"] = ram_gb()
    info["free_disk_gb"] = round(shutil.disk_usage(ROOT).free / 2**30, 1)
    info["nvidia_gpu"] = bool(shutil.which("nvidia-smi"))
    info["apple_silicon"] = info["os"] == "macos" and info["arch"] in ("arm64", "aarch64")
    return info


def ram_gb() -> float | None:
    try:
        if os_kind() == "linux":
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 2**20, 1)
        if os_kind() == "macos":
            return round(int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()) / 2**30, 1)
        if os_kind() == "windows":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = MS(); ms.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return round(ms.ullTotalPhys / 2**30, 1)
    except Exception:
        pass
    return None


def venv_dir(custom: str | None = None) -> Path:
    return Path(custom or os.environ.get("VENV") or ROOT / VENV_NAME).resolve()


def venv_python(venv: Path | None = None) -> Path:
    v = venv or venv_dir()
    return v / ("Scripts/python.exe" if os_kind() == "windows" else "bin/python")


def is_py311(exe: str) -> bool:
    try:
        out = subprocess.check_output([exe, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"], text=True,
                                      stderr=subprocess.DEVNULL, timeout=20).strip()
        return out == "%d.%d" % PY_REQUIRED
    except Exception:
        return False


def run(cmd: list[str], *, check: bool = True, env: dict | None = None, cwd: Path | None = None, quiet: bool = False) -> int:
    """명령을 그대로 실행하고 출력을 화면에 보여 준다."""
    if not quiet:
        print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    r = subprocess.run([str(c) for c in cmd], cwd=str(cwd or ROOT), env=e)
    if check and r.returncode != 0:
        raise subprocess.CalledProcessError(r.returncode, cmd)
    return r.returncode


def runtime_env() -> dict:
    """앱·평가·시험을 돌릴 때 공통으로 설정하는 환경 변수."""
    sep = os.pathsep
    src = str(ROOT / "src")
    return {"PYTHONPATH": src + (sep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
            "PYTORCH_ENABLE_MPS_FALLBACK": "1", "TRANSFORMERS_NO_TF": "1", "USE_TF": "0", "PYTHONUTF8": "1",
            "TOKENIZERS_PARALLELISM": "false"}


# ---------------------------------------------------------------- Ollama
def ollama_up(timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def ollama_models() -> list[str]:
    import json
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=5) as r:
            return [m.get("name", "") for m in json.loads(r.read().decode("utf-8")).get("models", [])]
    except Exception:
        return []


def ollama_installed() -> str | None:
    p = shutil.which("ollama")
    if p:
        return p
    for cand in ("/usr/local/bin/ollama", "/opt/homebrew/bin/ollama", "/Applications/Ollama.app/Contents/Resources/ollama",
                 str(Path.home() / "AppData/Local/Programs/Ollama/ollama.exe")):
        if Path(cand).exists():
            return cand
    return None


def start_ollama(wait: int = 40) -> bool:
    """서버가 꺼져 있으면 `ollama serve` 를 백그라운드로 띄우고 응답을 기다린다."""
    if ollama_up():
        return True
    exe = ollama_installed()
    if not exe:
        return False
    log = Path(os.environ.get("TMPDIR") or os.environ.get("TEMP") or "/tmp") / "intellectagent_ollama.log"
    kw: dict = {}
    if os_kind() == "windows":
        kw["creationflags"] = 0x00000008 | 0x00000200          # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    try:
        subprocess.Popen([exe, "serve"], stdout=open(log, "ab"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kw)
    except OSError:
        return False
    for _ in range(wait):
        if ollama_up():
            return True
        time.sleep(1)
    return False


def has_model(name: str) -> bool:
    names = ollama_models()
    base = name if ":" in name else name + ":latest"
    return name in names or base in names


def ollama_install_hint() -> str:
    return {"macos": "brew install ollama   (또는 https://ollama.com/download 에서 앱 설치)",
            "linux": "curl -fsSL https://ollama.com/install.sh | sh",
            "windows": "winget install -e --id Ollama.Ollama   (또는 https://ollama.com/download)"}.get(os_kind(), "https://ollama.com/download")

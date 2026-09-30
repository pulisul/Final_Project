from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any


def _run(cmd: list[str], timeout: int = 10) -> dict[str, Any]:
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        return {
            "available": True,
            "pass": p.returncode == 0,
            "returncode": p.returncode,
            "stdout": p.stdout[-6000:],
            "stderr": p.stderr[-3000:],
        }
    except FileNotFoundError:
        return {"available": False, "pass": False, "error": f"command_not_found:{cmd[0]}"}
    except subprocess.TimeoutExpired:
        return {"available": True, "pass": False, "timeout": True}


def disk_snapshot(root: Path, min_free_gb: float = 20.0) -> dict[str, Any]:
    usage = shutil.disk_usage(root)
    free_gb = usage.free / (1024 ** 3)
    return {
        "total_gb": round(usage.total / (1024 ** 3), 2),
        "used_gb": round(usage.used / (1024 ** 3), 2),
        "free_gb": round(free_gb, 2),
        "min_free_gb": min_free_gb,
        "pass": free_gb >= min_free_gb,
    }


def port_snapshot(host: str = "127.0.0.1", port: int = 11434, timeout: float = 0.5) -> dict[str, Any]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return {"host": host, "port": port, "open": True, "pass": True}
    except OSError as exc:
        return {"host": host, "port": port, "open": False, "pass": False, "error": f"{type(exc).__name__}: {exc}"}


def nvidia_snapshot() -> dict[str, Any]:
    q = _run([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ])
    if not q.get("available") or not q.get("pass"):
        return {"available": bool(q.get("available")), "pass": False, "raw": q}
    gpus = []
    for line in q.get("stdout", "").splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 4:
            name, driver, total, free = parts[:4]
            try:
                total_mib = int(total); free_mib = int(free)
            except ValueError:
                total_mib = free_mib = None
            gpus.append({"name": name, "driver": driver, "memory_total_mib": total_mib, "memory_free_mib": free_mib})
    return {"available": True, "pass": len(gpus) > 0, "gpus": gpus}


def torch_snapshot() -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {"installed": False, "pass": False, "error": "torch_not_installed"}
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        devices = []
        if cuda_available:
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                devices.append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_gb": round(p.total_memory / (1024 ** 3), 2),
                })
        return {
            "installed": True,
            "torch_version": getattr(torch, "__version__", None),
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "cuda_available": cuda_available,
            "device_count": len(devices),
            "devices": devices,
            # CPU-only torch is not itself an installation error. GPU-specific strict
            # validation decides whether CUDA is required for the deployment target.
            "pass": True,
        }
    except Exception as exc:
        return {"installed": True, "pass": False, "error": f"{type(exc).__name__}: {exc}"}


def ollama_model_snapshot(model: str) -> dict[str, Any]:
    q = _run(["ollama", "list"], timeout=15)
    if not q.get("available") or not q.get("pass"):
        return {"model": model, "installed": False, "pass": False, "raw": q}
    lines = q.get("stdout", "").splitlines()
    names = []
    for line in lines[1:]:
        if line.strip():
            names.append(line.split()[0])
    installed = model in names or any(n.startswith(model + ":") for n in names)
    return {"model": model, "installed": installed, "available_models": names, "pass": installed}


def diagnose(root: Path, model: str = "qwen2.5:14b", min_free_gb: float = 20.0) -> dict[str, Any]:
    disk = disk_snapshot(root, min_free_gb=min_free_gb)
    nvidia = nvidia_snapshot()
    torch = torch_snapshot()
    port = port_snapshot()
    ollama_model = ollama_model_snapshot(model)

    blockers: list[str] = []
    warnings: list[str] = []
    if not disk["pass"]:
        blockers.append("disk_space")
    if not torch.get("installed", False):
        blockers.append("torch_missing")
    elif not torch.get("pass", False):
        blockers.append("torch_broken")
    if nvidia.get("available") and torch.get("installed") and not torch.get("cuda_available", False):
        warnings.append("nvidia_detected_but_torch_cuda_unavailable")
    if not port.get("pass", False):
        warnings.append("ollama_port_closed")
    if not ollama_model.get("pass", False):
        warnings.append("ollama_model_missing_or_ollama_unavailable")

    return {
        "root": str(root),
        "model": model,
        "disk": disk,
        "nvidia": nvidia,
        "torch": torch,
        "ollama_port": port,
        "ollama_model": ollama_model,
        "blockers": blockers,
        "warnings": warnings,
        "pass": len(blockers) == 0,
    }


def classify_stage_failure(stage: str, result: dict[str, Any]) -> dict[str, Any]:
    text = (result.get("stderr", "") + "\n" + result.get("stdout", "")).lower()
    categories: list[str] = []
    if result.get("timeout"):
        categories.append("timeout")
    if "modulenotfounderror" in text or "no module named" in text:
        categories.append("missing_python_dependency")
    if "cuda" in text and any(x in text for x in ("not available", "no kernel image", "driver", "cudnn", "out of memory")):
        categories.append("cuda_or_gpu")
    if "connection refused" in text or "failed to connect" in text:
        categories.append("service_unreachable")
    if "no space left" in text:
        categories.append("disk_space")
    if "requirements-lock" in text or "python_311" in text or "pip check" in text:
        categories.append("environment_preflight")
    if not categories:
        categories.append("stage_specific_failure")
    return {"stage": stage, "categories": categories}

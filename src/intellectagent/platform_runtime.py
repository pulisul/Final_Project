from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class RuntimeProfile:
    profile: str
    os_name: str
    architecture: str
    accelerator: str
    torch_device: str
    is_apple_silicon: bool
    mps_available: bool
    cuda_available: bool
    pass_runtime: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["pass"] = d.pop("pass_runtime")
        return d


def _torch_flags() -> tuple[bool, bool, str | None]:
    try:
        import torch
    except Exception:
        return False, False, None
    cuda = bool(torch.cuda.is_available())
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return cuda, mps, getattr(torch, "__version__", None)


def detect_runtime_profile(requested: str = "auto") -> RuntimeProfile:
    os_name = platform.system()
    arch = platform.machine().lower()
    cuda, mps, _ = _torch_flags()
    apple = os_name == "Darwin" and arch in {"arm64", "aarch64"}

    if requested == "auto":
        if os_name == "Darwin":
            requested = "mac-native"
        elif cuda:
            requested = "linux-cuda"
        else:
            requested = "cpu"

    if requested == "mac-native":
        if os_name != "Darwin":
            return RuntimeProfile(requested, os_name, arch, "none", "cpu", apple, mps, cuda, False, "mac_native_requires_macos")
        accelerator = "mps" if mps else "cpu"
        device = "mps" if mps else "cpu"
        return RuntimeProfile(requested, os_name, arch, accelerator, device, apple, mps, cuda, True, "macos_supported")

    if requested == "linux-cuda":
        ok = cuda
        return RuntimeProfile(requested, os_name, arch, "cuda" if cuda else "none", "cuda" if cuda else "cpu", apple, mps, cuda, ok, "cuda_available" if ok else "cuda_required")

    if requested == "cpu":
        return RuntimeProfile(requested, os_name, arch, "cpu", "cpu", apple, mps, cuda, True, "cpu_profile")

    raise ValueError(f"unknown runtime profile: {requested}")


def memory_snapshot(profile: RuntimeProfile) -> dict[str, Any]:
    out: dict[str, Any] = {
        "profile": profile.profile,
        "accelerator": profile.accelerator,
        "torch_device": profile.torch_device,
    }
    try:
        import torch
        if profile.accelerator == "cuda" and torch.cuda.is_available():
            out.update({
                "available": True,
                "allocated_mib": round(torch.cuda.memory_allocated() / (1024**2), 2),
                "reserved_mib": round(torch.cuda.memory_reserved() / (1024**2), 2),
            })
        elif profile.accelerator == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            allocated = getattr(torch.mps, "current_allocated_memory", lambda: 0)()
            driver = getattr(torch.mps, "driver_allocated_memory", lambda: 0)()
            out.update({
                "available": True,
                "allocated_mib": round(allocated / (1024**2), 2),
                "driver_allocated_mib": round(driver / (1024**2), 2),
            })
        else:
            out["available"] = profile.accelerator == "cpu"
    except Exception as exc:
        out.update({"available": False, "error": f"{type(exc).__name__}: {exc}"})

    if platform.system() == "Darwin":
        try:
            p = subprocess.run(["sysctl", "-n", "hw.memsize"], text=True, capture_output=True, timeout=3)
            if p.returncode == 0:
                out["unified_memory_gb"] = round(int(p.stdout.strip()) / (1024**3), 2)
        except Exception:
            pass
    return out


def configure_mac_mps_fallback(profile: RuntimeProfile) -> None:
    if profile.profile == "mac-native":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Sequence


def _run(cmd: list[str], timeout: int = 20) -> dict[str, Any]:
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        return {"pass": p.returncode == 0, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except FileNotFoundError:
        return {"pass": False, "error": f"command_not_found:{cmd[0]}"}
    except subprocess.TimeoutExpired:
        return {"pass": False, "timeout": True}


def gpu_memory_snapshot() -> dict[str, Any]:
    q = _run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    if not q.get("pass"):
        return {"available": False, "pass": False, "raw": q}
    rows = []
    for line in q["stdout"].splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 6:
            continue
        idx, name, total, used, free, util = parts[:6]
        try:
            rows.append({
                "index": int(idx), "name": name,
                "memory_total_mib": int(total), "memory_used_mib": int(used),
                "memory_free_mib": int(free), "utilization_gpu_pct": int(util),
            })
        except ValueError:
            continue
    return {"available": True, "pass": bool(rows), "gpus": rows}


def torch_cuda_snapshot(require_cuda: bool = True) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"installed": False, "pass": False, "error": f"{type(exc).__name__}: {exc}"}
    cuda = bool(torch.cuda.is_available())
    devices = []
    if cuda:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            devices.append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(p.total_memory / (1024**3), 2),
                "capability": list(torch.cuda.get_device_capability(i)),
            })
    return {
        "installed": True,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda,
        "device_count": len(devices),
        "devices": devices,
        "pass": cuda if require_cuda else True,
    }


def finite_nonzero_vectors(vectors: Sequence[Sequence[float]]) -> bool:
    return bool(vectors) and all(
        len(v) > 0 and all(math.isfinite(float(x)) for x in v) and any(abs(float(x)) > 1e-12 for x in v)
        for v in vectors
    )


@dataclass(frozen=True)
class CandidateResult:
    candidate_k: int
    hit_rate_at_k: float
    mrr: float
    latency_ms_mean: float
    latency_ms_p95: float
    search_ms_mean: float
    rerank_ms_mean: float
    reranker_fallback_rate: float


def choose_candidate(results: Sequence[CandidateResult]) -> CandidateResult:
    if not results:
        raise ValueError("results must not be empty")
    # Quality first. If quality is identical, prefer lower latency, then smaller candidate_k.
    return sorted(results, key=lambda r: (-r.hit_rate_at_k, -r.mrr, r.latency_ms_mean, r.candidate_k))[0]




def candidate_quality_pass(result: CandidateResult, *, min_hit_rate_at_k: float, min_mrr: float) -> bool:
    return result.hit_rate_at_k >= float(min_hit_rate_at_k) and result.mrr >= float(min_mrr)

def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

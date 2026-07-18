from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

# Primary batch-size-1 claim methods. Batched dynamic regrouping is intentionally
# excluded and must remain a separate experiment.
PRIMARY_METHODS: tuple[str, ...] = (
    "equal_full_depth",
    "dynamic_fusion_full_depth",
    "fixed_12",
    "fixed_18",
    "adaptive_conservative",
    "adaptive_balanced",
    "adaptive_aggressive",
)

SEGMENT_KEYS: tuple[str, ...] = (
    "backbone",
    "maps",
    "descriptors",
    "dlcm",
    "lse",
    "postprocess",
    "total",
)


def percentile_stats(samples: list[float] | np.ndarray) -> dict[str, float]:
    xs = np.asarray(list(samples), dtype=np.float64).reshape(-1)
    if xs.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }
    return {
        "n": int(xs.size),
        "mean": float(np.mean(xs)),
        "median": float(np.percentile(xs, 50)),
        "p90": float(np.percentile(xs, 90)),
        "p95": float(np.percentile(xs, 95)),
    }


def run_cuda_event_timings(
    fn: Callable[[], Any],
    *,
    device: torch.device,
    warmup: int,
    repetitions: int,
) -> list[float]:
    """Time ``fn`` with CUDA events; synchronize around each measured segment."""
    if device.type != "cuda":
        raise ValueError("run_cuda_event_timings requires a CUDA device")
    for _ in range(max(0, int(warmup))):
        fn()
    torch.cuda.synchronize(device)
    times_ms: list[float] = []
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    for _ in range(max(1, int(repetitions))):
        torch.cuda.synchronize(device)
        starter.record()
        fn()
        ender.record()
        torch.cuda.synchronize(device)
        times_ms.append(float(starter.elapsed_time(ender)))
    return times_ms


def _run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ""


def collect_device_metadata(
    *,
    device: torch.device,
    image_size: int,
    dtype: str,
) -> dict[str, Any]:
    gpu_model = None
    driver_version = None
    cuda_runtime = getattr(torch.version, "cuda", None)
    memory: dict[str, Any] = {}
    clocks: dict[str, Any] = {}

    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index or 0
        gpu_model = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        memory = {
            "total_mb": float(props.total_memory) / (1024.0 * 1024.0),
            "allocated_mb": float(torch.cuda.memory_allocated(idx)) / (1024.0 * 1024.0),
            "reserved_mb": float(torch.cuda.memory_reserved(idx)) / (1024.0 * 1024.0),
        }
        smi = _run_cmd(
            [
                "nvidia-smi",
                f"--id={idx}",
                "--query-gpu=driver_version,clocks.sm,clocks.mem,clocks.gr",
                "--format=csv,noheader,nounits",
            ]
        )
        if smi:
            parts = [p.strip() for p in smi.split(",")]
            if parts:
                driver_version = parts[0]
            if len(parts) >= 4:
                clocks = {
                    "sm_mhz": parts[1],
                    "mem_mhz": parts[2],
                    "graphics_mhz": parts[3],
                }

    nvcc_out = _run_cmd(["nvcc", "--version"])
    nvcc_version = None
    m = re.search(r"release\s+([0-9.]+)", nvcc_out)
    if m:
        nvcc_version = m.group(1)
    elif nvcc_out:
        nvcc_version = nvcc_out.splitlines()[-1]

    return {
        "gpu_model": gpu_model,
        "driver_version": driver_version,
        "cuda_version": cuda_runtime,
        "nvcc_version": nvcc_version,
        "torch_version": torch.__version__,
        "device": str(device),
        "image_resolution": int(image_size),
        "dtype": str(dtype),
        "memory": memory,
        "clocks": clocks,
    }


def empty_segment_accumulator() -> dict[str, list[float]]:
    return {k: [] for k in SEGMENT_KEYS}


def summarize_segments(acc: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    return {k: percentile_stats(v) for k, v in acc.items()}

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rad.evaluation.benchmark import (
    PRIMARY_METHODS,
    collect_device_metadata,
    percentile_stats,
    run_cuda_event_timings,
)


def test_percentile_stats_hand_fixture():
    xs = [float(i) for i in range(1, 101)]
    stats = percentile_stats(xs)
    assert set(stats) >= {"median", "mean", "p90", "p95", "n"}
    assert stats["n"] == 100
    assert stats["mean"] == pytest.approx(50.5)
    assert stats["median"] == pytest.approx(50.5)
    assert stats["p90"] == pytest.approx(90.1)
    assert stats["p95"] == pytest.approx(95.05)


def test_primary_methods_exclude_batched_regrouping():
    assert "batched_dynamic_regrouping" not in PRIMARY_METHODS
    required = {
        "equal_full_depth",
        "dynamic_fusion_full_depth",
        "fixed_12",
        "fixed_18",
        "adaptive_conservative",
        "adaptive_balanced",
        "adaptive_aggressive",
    }
    assert required.issubset(set(PRIMARY_METHODS))


def test_collect_device_metadata_has_required_keys():
    meta = collect_device_metadata(
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        image_size=32,
        dtype="float32",
    )
    for key in (
        "gpu_model",
        "driver_version",
        "cuda_version",
        "nvcc_version",
        "image_resolution",
        "dtype",
        "memory",
    ):
        assert key in meta


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_event_timings_smoke():
    device = torch.device("cuda:0")
    x = torch.randn(1, 3, 32, 32, device=device)

    def fn():
        y = x * 2
        return y.sum()

    times = run_cuda_event_timings(fn, device=device, warmup=2, repetitions=5)
    assert len(times) == 5
    assert all(t >= 0.0 for t in times)
    stats = percentile_stats(times)
    assert stats["n"] == 5


def test_benchmark_cli_dry_run(tmp_path: Path):
    """Smoke: CLI dry-run exits 0 and does not claim batched regrouping as primary."""
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[2]
    cfg = repo / "configs" / "rad" / "benchmark.yaml"
    assert cfg.is_file()
    proc = subprocess.run(
        [
            sys.executable,
            str(repo / "tools" / "benchmark_latency.py"),
            "--config",
            str(cfg),
            "--dry-run",
            "--output-dir",
            str(tmp_path / "bench"),
        ],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "batched_dynamic_regrouping" not in proc.stdout.lower() or "separate" in proc.stdout.lower()
    assert "warmup" in proc.stdout.lower()
    assert "repetitions" in proc.stdout.lower() or "reps" in proc.stdout.lower()

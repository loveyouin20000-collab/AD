from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


def _run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def test_smoke_cli_dry_run_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "smoke_out"
    proc = _run(
        [
            PYTHON,
            "tools/smoke_adaptive_engine.py",
            "--config",
            "configs/rad/adaptive.yaml",
            "--seed",
            "111",
            "--output-dir",
            str(out),
            "--dry-run",
        ]
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "dry-run" in (proc.stdout + proc.stderr).lower()
    assert not out.exists()


def test_evaluate_adaptive_dataset_dry_run_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "dataset_out"
    proc = _run(
        [
            PYTHON,
            "tools/evaluate_adaptive_dataset.py",
            "--config",
            "configs/rad/adaptive.yaml",
            "--seed",
            "111",
            "--output-dir",
            str(out),
            "--dry-run",
        ]
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    blob = proc.stdout + proc.stderr
    assert "dry-run" in blob.lower()
    assert "adapter" in blob.lower() or "dataset" in blob.lower()
    assert not out.exists()


def test_evaluate_adaptive_dataset_has_no_synthetic_or_visa_parser() -> None:
    src = (REPO_ROOT / "tools" / "evaluate_adaptive_dataset.py").read_text(encoding="utf-8")
    assert "_synthetic_batch" not in src
    assert "torch.randn" not in src
    assert "_load_visa_index" not in src
    assert "evaluate_dataset" in src
    assert "compute_paper_metrics" in src
    assert "get_adapter" in src


def test_smoke_cli_forbids_paper_metrics_in_source() -> None:
    src = (REPO_ROOT / "tools" / "smoke_adaptive_engine.py").read_text(encoding="utf-8")
    assert "compute_paper_metrics" not in src
    assert "PaperMetrics" not in src
    assert "_synthetic_batch" in src or "randn" in src


def test_legacy_evaluate_adaptive_forwards_to_smoke() -> None:
    src = (REPO_ROOT / "tools" / "evaluate_adaptive.py").read_text(encoding="utf-8")
    assert "smoke_adaptive_engine" in src


def test_zero_shot_transfer_uses_adapter_not_visa_parser() -> None:
    src = (REPO_ROOT / "tools" / "evaluate_zero_shot_transfer.py").read_text(
        encoding="utf-8"
    )
    assert "_load_visa_index" not in src
    assert "get_adapter" in src
    assert "evaluate_dataset" in src
    assert "compute_paper_metrics" in src
    assert "forbid_target_access_during_calibration" in src
    assert "assert_policy_unchanged" in src
    assert "pro_score_proxy" not in src


def test_zero_shot_dry_run_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "zs_out"
    proc = _run(
        [
            PYTHON,
            "tools/evaluate_zero_shot_transfer.py",
            "--config",
            "configs/rad/zero_shot_transfer.yaml",
            "--seed",
            "111",
            "--output-dir",
            str(out),
            "--dry-run",
        ]
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    blob = proc.stdout + proc.stderr
    assert "dry-run" in blob.lower()
    assert "policy_digest" in blob
    assert not out.exists()

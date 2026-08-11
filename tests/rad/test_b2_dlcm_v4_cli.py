"""V4 CLI tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable


def test_cli_dry_run_prints_flags() -> None:
    proc = subprocess.run(
        [
            PY,
            str(REPO / "tools/train_b2_dlcm_v4.py"),
            "--config",
            str(REPO / "configs/phase_b/b2_dlcm_uniform_relative_contract_v4.json"),
            "--seed",
            "17",
            "--output-dir",
            "/tmp/v4-cli-dry-should-not-exist",
            "--dry-run",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    for line in (
        "real_training_started = false",
        "development_evaluation_started = false",
        "final_content_resolved = false",
        "final_materialization_started = false",
        "final_evaluation_started = false",
        "artifact_written = false",
        "run_directory_created = false",
        "teacher_forward_count = 0",
    ):
        assert line in out

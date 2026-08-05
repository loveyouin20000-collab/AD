"""V5 CLI dry-run tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = "/root/miniconda3/envs/rad-visualad/bin/python"


def test_cli_dry_run_prints_flags() -> None:
    proc = subprocess.run(
        [
            PY,
            str(REPO / "tools/calibrate_b2_dlcm_v5.py"),
            "--config",
            str(REPO / "configs/phase_b/b2_dlcm_uniform_anchored_contract_v5.json"),
            "--output-dir",
            "/tmp/v5-cli-dry-should-not-exist",
            "--dry-run",
            "--process-label",
            "A",
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
        "calibration_started = false",
        "development_evaluation_started = false",
        "final_content_resolved = false",
        "final_materialization_started = false",
        "final_evaluation_started = false",
        "artifact_written = false",
        "run_directory_created = false",
        "teacher_forward_count = 0",
    ):
        assert line in out
    assert not Path("/tmp/v5-cli-dry-should-not-exist").exists()

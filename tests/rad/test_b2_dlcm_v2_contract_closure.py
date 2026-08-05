"""Contract closure: dry-run twice with argument permutation."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = "/root/miniconda3/envs/rad-visualad/bin/python"
CONFIG = REPO / "configs/phase_b/b2_dlcm_decoupled_training_contract_v2.json"


def _dry(args: list[str]) -> str:
    proc = subprocess.run(
        [PYTHON, str(REPO / "tools/train_b2_dlcm_v2.py"), *args],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
        env={
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "PYTHONPATH": str(REPO),
        },
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_dry_run_twice_argument_permutation(tmp_path: Path) -> None:
    out1 = tmp_path / "a"
    out2 = tmp_path / "b"
    stdout1 = _dry(
        ["--config", str(CONFIG), "--seed", "17", "--output-dir", str(out1), "--dry-run"]
    )
    stdout2 = _dry(
        ["--seed", "17", "--dry-run", "--output-dir", str(out2), "--config", str(CONFIG)]
    )
    required = [
        "real_training_started = false",
        "development_evaluation_started = false",
        "final_content_resolved = false",
        "final_materialization_started = false",
        "final_evaluation_started = false",
        "artifact_written = false",
        "run_directory_created = false",
        "teacher_forward_count = 0",
    ]
    for line in required:
        assert line in stdout1
        assert line in stdout2
    assert not out1.exists()
    assert not out2.exists()
    assert "B2_DLCM_V2_TRAINING_RESULT=" in stdout1
    assert "B2_DLCM_V2_TRAINING_RESULT=" in stdout2

import subprocess
import sys
from pathlib import Path

from tests.rad.contracts.baseline import (
    assert_checkpoint_dry_run_skips_train,
    assert_dry_run_resolves_train_and_test,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


def _run_baseline_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PYTHON, "tools/reproduce_baseline.py", *args],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )


def test_baseline_dry_run_resolves_paths_and_command():
    result = _run_baseline_cli(
        [
            "--config",
            "configs/rad/baseline_mvtec_to_visa.yaml",
            "--dry-run",
        ]
    )
    assert_dry_run_resolves_train_and_test(result.stdout)


def test_baseline_checkpoint_dry_run_skips_train():
    checkpoint = "weight/train_on_mvtec/CLIP.pth"
    result = _run_baseline_cli(
        [
            "--config",
            "configs/rad/baseline_mvtec_to_visa.yaml",
            "--checkpoint",
            checkpoint,
            "--dry-run",
        ]
    )
    assert_checkpoint_dry_run_skips_train(result.stdout, checkpoint)

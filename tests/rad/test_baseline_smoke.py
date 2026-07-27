import subprocess
import sys
from pathlib import Path

from tests.rad.contracts.baseline import assert_baseline_dry_run_contract

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


def test_baseline_dry_run_resolves_paths_and_command(tmp_path: Path):
    output_dir = tmp_path / "baseline_smoke_out"
    train_data = tmp_path / "mvtec"
    test_data = tmp_path / "visa"
    train_data.mkdir()
    test_data.mkdir()
    result = _run_baseline_cli(
        [
            "--config",
            "configs/rad/baseline_mvtec_to_visa.yaml",
            "--train-data-path",
            str(train_data),
            "--test-data-path",
            str(test_data),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )
    assert_baseline_dry_run_contract(result.stdout, output_dir=output_dir)


def test_baseline_checkpoint_dry_run_skips_train(tmp_path: Path):
    checkpoint = "weight/train_on_mvtec/CLIP.pth"
    output_dir = tmp_path / "baseline_ckpt_out"
    train_data = tmp_path / "mvtec"
    test_data = tmp_path / "visa"
    train_data.mkdir()
    test_data.mkdir()
    result = _run_baseline_cli(
        [
            "--config",
            "configs/rad/baseline_mvtec_to_visa.yaml",
            "--train-data-path",
            str(train_data),
            "--test-data-path",
            str(test_data),
            "--checkpoint",
            checkpoint,
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )
    assert_baseline_dry_run_contract(
        result.stdout,
        checkpoint=checkpoint,
        output_dir=output_dir,
    )

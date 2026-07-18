import subprocess


def test_baseline_dry_run_resolves_paths_and_command():
    result = subprocess.run(
        [
            "python",
            "tools/reproduce_baseline.py",
            "--config",
            "configs/rad/baseline_mvtec_to_visa.yaml",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "train.py" in result.stdout
    assert "features_list 6 12 18 24" in result.stdout


def test_baseline_checkpoint_dry_run_skips_train():
    result = subprocess.run(
        [
            "python",
            "tools/reproduce_baseline.py",
            "--config",
            "configs/rad/baseline_mvtec_to_visa.yaml",
            "--checkpoint",
            "weight/train_on_mvtec/CLIP.pth",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "SKIPPED (using --checkpoint)" in result.stdout
    assert "weight/train_on_mvtec/CLIP.pth" in result.stdout
    assert "test.py" in result.stdout

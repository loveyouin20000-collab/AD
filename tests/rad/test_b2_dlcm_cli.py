"""CLI contract tests for B2-05A (dry-run only; no fixture flag)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = "/root/miniconda3/envs/rad-visualad/bin/python"
CLI = REPO / "tools" / "train_b2_dlcm.py"
VERIFY = REPO / "tools" / "verify_b2_dlcm_artifacts.py"
CONFIG = REPO / "configs" / "phase_b" / "b2_dlcm_training_contract_v1.json"


def _run(args: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    base_env = {
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
        "CUDA_VISIBLE_DEVICES": "",
        "PATH": str(Path(sys.executable).parent),
    }
    if env:
        base_env.update(env)
    return subprocess.run(
        [PYTHON, *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**dict(**{k: v for k, v in __import__("os").environ.items()}), **base_env},
        check=False,
    )


def test_dry_run_twice_permutation_no_writes(tmp_path: Path) -> None:
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    common = [
        str(CLI),
        "--config",
        str(CONFIG),
        "--descriptor-manifest",
        str(tmp_path / "desc.json"),
        "--descriptor-root",
        str(tmp_path / "desc"),
        "--contribution-target-manifest",
        str(tmp_path / "tgt.json"),
        "--contribution-target-root",
        str(tmp_path / "tgt"),
        "--output-root",
        str(out_a),
        "--seed",
        "17",
        "--dry-run",
    ]
    for path in (
        tmp_path / "desc.json",
        tmp_path / "tgt.json",
    ):
        path.write_text("{}", encoding="utf-8")
    (tmp_path / "desc").mkdir()
    (tmp_path / "tgt").mkdir()

    first = _run(common)
    assert first.returncode == 0, first.stderr
    assert "artifact_written = false" in first.stdout
    assert "run_directory_created = false" in first.stdout
    assert "real_training_started = false" in first.stdout
    assert "evaluation_unlocked = false" in first.stdout
    assert "teacher_forward_count = 0" in first.stdout
    assert not out_a.exists()

    permuted = [
        str(CLI),
        "--seed",
        "17",
        "--dry-run",
        "--output-root",
        str(out_b),
        "--contribution-target-root",
        str(tmp_path / "tgt"),
        "--contribution-target-manifest",
        str(tmp_path / "tgt.json"),
        "--descriptor-root",
        str(tmp_path / "desc"),
        "--descriptor-manifest",
        str(tmp_path / "desc.json"),
        "--config",
        str(CONFIG),
    ]
    second = _run(permuted)
    assert second.returncode == 0, second.stderr
    assert not out_b.exists()


def test_non_dry_run_disabled() -> None:
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        (t / "desc.json").write_text("{}", encoding="utf-8")
        (t / "tgt.json").write_text("{}", encoding="utf-8")
        (t / "desc").mkdir()
        (t / "tgt").mkdir()
        proc = _run(
            [
                str(CLI),
                "--config",
                str(CONFIG),
                "--descriptor-manifest",
                str(t / "desc.json"),
                "--descriptor-root",
                str(t / "desc"),
                "--contribution-target-manifest",
                str(t / "tgt.json"),
                "--contribution-target-root",
                str(t / "tgt"),
                "--output-root",
                str(t / "out"),
                "--seed",
                "17",
            ]
        )
    assert proc.returncode != 0
    assert "B2_DLCM_REAL_TRAINING_NOT_ENABLED" in (proc.stdout + proc.stderr)


def test_no_fixture_flag_exposed() -> None:
    help_proc = _run([str(CLI), "--help"])
    assert "--fixture" not in help_proc.stdout
    assert "--skip-identity" not in help_proc.stdout
    assert "--evaluation-unlock" not in help_proc.stdout


def test_verify_cli_help() -> None:
    proc = _run([str(VERIFY), "--help"])
    assert proc.returncode == 0
    assert "--config" in proc.stdout

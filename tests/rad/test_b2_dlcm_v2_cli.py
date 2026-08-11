"""RED/GREEN tests for B2-05C1 V2 CLIs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
CONFIG = REPO / "configs/phase_b/b2_dlcm_decoupled_training_contract_v2.json"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
        env={
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(REPO),
        },
    )


def test_train_cli_dry_run(tmp_path: Path) -> None:
    out = tmp_path / "dry"
    proc = _run(
        [
            PYTHON,
            str(REPO / "tools/train_b2_dlcm_v2.py"),
            "--config",
            str(CONFIG),
            "--seed",
            "17",
            "--output-dir",
            str(out),
            "--dry-run",
        ]
    )
    assert proc.returncode == 0, proc.stderr
    for field in (
        "real_training_started = false",
        "development_evaluation_started = false",
        "final_content_resolved = false",
        "final_materialization_started = false",
        "final_evaluation_started = false",
        "artifact_written = false",
        "run_directory_created = false",
        "teacher_forward_count = 0",
    ):
        assert field in proc.stdout
    assert not out.exists()


def test_train_cli_without_dry_run_fails() -> None:
    proc = _run(
        [
            PYTHON,
            str(REPO / "tools/train_b2_dlcm_v2.py"),
            "--config",
            str(CONFIG),
            "--seed",
            "17",
            "--output-dir",
            "/tmp/should_not_exist_v2",
        ]
    )
    assert proc.returncode != 0
    assert "B2_DLCM_V2_REAL_TRAINING_NOT_ENABLED" in proc.stderr


def test_materialize_and_evaluate_locked(tmp_path: Path) -> None:
    roster = tmp_path / "roster.json"
    roster.write_text("{}", encoding="utf-8")
    mat = _run(
        [
            PYTHON,
            str(REPO / "tools/materialize_b2_dlcm_final_v2.py"),
            "--roster",
            str(roster),
            "--output-dir",
            str(tmp_path / "mat"),
            "--dry-run",
        ]
    )
    assert mat.returncode != 0
    assert "B2_DLCM_FINAL_MATERIALIZATION_UNLOCK_REQUIRED" in mat.stderr

    ev = _run(
        [
            PYTHON,
            str(REPO / "tools/evaluate_b2_dlcm_final_v2.py"),
            "--materialization-manifest",
            str(roster),
            "--output-dir",
            str(tmp_path / "ev"),
            "--dry-run",
        ]
    )
    assert ev.returncode != 0
    assert "B2_DLCM_FINAL_EVALUATION_UNLOCK_REQUIRED" in ev.stderr


def test_verify_artifacts_cli(tmp_path: Path) -> None:
    from rad.phase_b import b2_dlcm_v2_protocol as protocol

    art = tmp_path / "aux.json"
    payload = {
        "schema_version": "b2_dlcm_v2_auxiliary_diagnostics_v1",
        "ok": True,
    }
    protocol.persist_json_atomic(art, payload)
    proc = _run(
        [
            PYTHON,
            str(REPO / "tools/verify_b2_dlcm_v2_artifacts.py"),
            "--artifact",
            str(art),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    body = json.loads(proc.stdout)
    assert body["status"] == "ok"

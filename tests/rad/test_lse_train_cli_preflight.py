from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import tools.train_lse as train_lse


def test_legacy_lse_dry_run_requires_b2_accepted_gate() -> None:
    with pytest.raises(SystemExit) as exc:
        train_lse.main(["--config", "configs/rad/lse.yaml", "--dry-run", "--device", "cpu"])
    assert exc.value.code == 2


def test_b2_lse_dry_run_requires_training_unlock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        train_lse.accepted_gate,
        "run_lse_preflight",
        lambda _cfg: {
            "ready": True,
            "accepted_gate_passed": True,
            "training_started": False,
            "accepted_identity": "accepted-id",
            "v5_deployment_identity": "deploy-id",
            "H_decision": "decision-id",
            "H_evidence": "evidence-id",
            "dlcm_checkpoint": "checkpoint.pt",
            "missing_prerequisites": [],
        },
    )

    with pytest.raises(SystemExit) as exc:
        train_lse.main(
            [
                "--config",
                "configs/rad/lse_b2_accepted_v5.yaml",
                "--dry-run",
                "--device",
                "cpu",
                "--output-dir",
                "artifacts/checkpoints/lse/b2_06d_first_controlled_run",
            ]
        )

    assert exc.value.code == 2


def test_b2_lse_dry_run_does_not_hash_checkpoint_before_unlock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = yaml.safe_load(
        Path("configs/rad/lse_b2_accepted_v5.yaml").read_text(encoding="utf-8")
    )
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint")
    config["lse"]["dlcm_checkpoint"] = str(checkpoint)
    config["lse"].pop("training_unlock_manifest")
    config_path = tmp_path / "lse.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    monkeypatch.setattr(
        train_lse.accepted_gate,
        "run_lse_preflight",
        lambda _cfg: {
            "ready": True,
            "accepted_gate_passed": True,
            "training_started": False,
            "accepted_identity": "accepted-id",
            "v5_deployment_identity": "deploy-id",
            "H_decision": "decision-id",
            "H_evidence": "evidence-id",
            "dlcm_checkpoint": str(checkpoint),
            "missing_prerequisites": [],
        },
    )

    def fail_if_checkpoint_is_hashed(path: Path) -> str:
        if Path(path) == checkpoint:
            raise AssertionError("checkpoint must not be hashed before unlock validation")
        return "config-hash"

    monkeypatch.setattr(train_lse, "sha256_file", fail_if_checkpoint_is_hashed)

    with pytest.raises(SystemExit) as exc:
        train_lse.main(
            [
                "--config",
                str(config_path),
                "--dry-run",
                "--device",
                "cpu",
                "--output-dir",
                "artifacts/checkpoints/lse/b2_06d_first_controlled_run",
            ]
        )

    assert exc.value.code == 2
    assert "B2_LSE_TRAINING_UNLOCK_REQUIRED" in capsys.readouterr().err

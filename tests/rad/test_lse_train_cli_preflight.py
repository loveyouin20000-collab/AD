from __future__ import annotations

import pytest

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

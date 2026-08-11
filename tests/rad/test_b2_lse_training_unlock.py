from __future__ import annotations

import json
from pathlib import Path

import pytest

from rad.phase_b import b2_lse_training_unlock as unlock

REPO = Path(__file__).resolve().parents[2]
FROZEN_CONFIG = REPO / "configs" / "rad" / "lse_b2_accepted_v5.yaml"
FROZEN_UNLOCK = REPO / "docs" / "phase_b" / "b2_06d_lse_training_unlock.json"


def _base_preflight() -> dict[str, object]:
    return {
        "ready": True,
        "accepted_gate_passed": True,
        "training_started": False,
        "accepted_identity": "accepted-id",
        "v5_deployment_identity": "deploy-id",
        "H_decision": "decision-id",
        "H_evidence": "evidence-id",
    }


def test_frozen_accepted_lse_config_hash_matches_training_unlock() -> None:
    payload = json.loads(FROZEN_UNLOCK.read_text(encoding="utf-8"))
    assert unlock.sha256_file(FROZEN_CONFIG) == payload["config_sha256"]


def _write_unlock(path: Path, *, output_dir: Path, accepted_identity: str = "accepted-id") -> None:
    payload = {
        "schema_version": "b2_06d_lse_training_unlock_v1",
        "purpose": "first_controlled_lse_training",
        "accepted_identity": accepted_identity,
        "v5_deployment_identity": "deploy-id",
        "H_decision": "decision-id",
        "H_evidence": "evidence-id",
        "config_sha256": "config-sha",
        "train_output_dir": str(output_dir),
        "seed": 111,
        "epochs": 30,
        "patience": 10,
        "training_started": False,
        "lse_checkpoint_generated": False,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_validate_training_unlock_accepts_matching_contract(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    path = tmp_path / "unlock.json"
    _write_unlock(path, output_dir=output_dir)

    report = unlock.validate_training_unlock(
        path,
        preflight=_base_preflight(),
        config_sha256="config-sha",
        train_output_dir=output_dir,
        seed=111,
        epochs=30,
        patience=10,
    )

    assert report["ready"] is True
    assert report["training_started"] is False
    assert report["unlock_identity"]


def test_validate_training_unlock_rejects_identity_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "unlock.json"
    _write_unlock(path, output_dir=tmp_path / "run", accepted_identity="wrong")

    with pytest.raises(unlock.B2LSETrainingUnlockError) as exc:
        unlock.validate_training_unlock(
            path,
            preflight=_base_preflight(),
            config_sha256="config-sha",
            train_output_dir=tmp_path / "run",
            seed=111,
            epochs=30,
            patience=10,
        )

    assert exc.value.code == "B2_LSE_TRAINING_UNLOCK_IDENTITY_MISMATCH"


def test_validate_training_unlock_rejects_reuse_after_receipt(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "b2_06d_lse_training_receipt.json").write_text("{}\n", encoding="utf-8")
    path = tmp_path / "unlock.json"
    _write_unlock(path, output_dir=output_dir)

    with pytest.raises(unlock.B2LSETrainingUnlockError) as exc:
        unlock.validate_training_unlock(
            path,
            preflight=_base_preflight(),
            config_sha256="config-sha",
            train_output_dir=output_dir,
            seed=111,
            epochs=30,
            patience=10,
        )

    assert exc.value.code == "B2_LSE_TRAINING_UNLOCK_ALREADY_CONSUMED"

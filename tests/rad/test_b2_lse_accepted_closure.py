from __future__ import annotations

import pytest

from rad.phase_b import b2_lse_accepted_closure as closure


def _decision() -> dict[str, object]:
    return {
        "schema_version": "b2_06e_lse_qualification_decision_v1",
        "verdict": "qualified",
        "H_lse_qualification": "qualification-id",
        "accepted_artifact_generated": False,
        "accepted_identity": "dlcm-accepted-id",
        "v5_deployment_identity": "deploy-id",
        "unlock_identity": "unlock-id",
        "training_receipt_identity": "receipt-id",
        "best_checkpoint_sha256": "lse-sha",
        "calibration_nll": 0.4,
        "max_calibration_nll": 0.5,
        "evaluated_rows": 16,
    }


def _receipt() -> dict[str, object]:
    return {
        "schema_version": "b2_06d_lse_training_receipt_v1",
        "receipt_identity": "receipt-id",
        "accepted_identity": "dlcm-accepted-id",
        "v5_deployment_identity": "deploy-id",
        "unlock_identity": "unlock-id",
        "best_checkpoint_sha256": "lse-sha",
        "training_started": True,
        "lse_checkpoint_generated": True,
    }


def test_build_accepted_lse_manifest_binds_qualified_decision() -> None:
    manifest = closure.build_accepted_lse_manifest(
        decision=_decision(),
        training_receipt=_receipt(),
        lse_checkpoint_sha256="lse-sha",
        accepted_checkpoint_path="accepted_refs/lse_best.pt",
        source_checkpoint_path="artifacts/checkpoints/lse/lse_best.pt",
        closure_git_sha="closure-git",
    )

    assert manifest["schema_version"] == "b2_06f_lse_accepted_artifact_manifest_v1"
    assert manifest["lse_qualified"] is True
    assert manifest["accepted_lse_identity"]
    assert manifest["training_started"] is False
    assert manifest["evaluation_started"] is False
    assert manifest["accepted_artifact_generated"] is True


def test_build_accepted_lse_manifest_rejects_unqualified_decision() -> None:
    decision = _decision()
    decision["verdict"] = "failed"

    with pytest.raises(closure.B2LSEAcceptedClosureError) as exc:
        closure.build_accepted_lse_manifest(
            decision=decision,
            training_receipt=_receipt(),
            lse_checkpoint_sha256="lse-sha",
            accepted_checkpoint_path="accepted_refs/lse_best.pt",
            source_checkpoint_path="artifacts/checkpoints/lse/lse_best.pt",
            closure_git_sha="closure-git",
        )

    assert exc.value.code == "B2_LSE_ACCEPTED_CLOSURE_NOT_QUALIFIED"


def test_build_accepted_lse_manifest_rejects_checkpoint_sha_mismatch() -> None:
    with pytest.raises(closure.B2LSEAcceptedClosureError) as exc:
        closure.build_accepted_lse_manifest(
            decision=_decision(),
            training_receipt=_receipt(),
            lse_checkpoint_sha256="wrong",
            accepted_checkpoint_path="accepted_refs/lse_best.pt",
            source_checkpoint_path="artifacts/checkpoints/lse/lse_best.pt",
            closure_git_sha="closure-git",
        )

    assert exc.value.code == "B2_LSE_ACCEPTED_CLOSURE_CHECKPOINT_MISMATCH"

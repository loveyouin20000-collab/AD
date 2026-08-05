from __future__ import annotations

import pytest

from rad.phase_b import b2_lse_qualification as qual


def _receipt() -> dict[str, object]:
    return {
        "schema_version": "b2_06d_lse_training_receipt_v1",
        "accepted_identity": "accepted-id",
        "v5_deployment_identity": "deploy-id",
        "H_decision": "decision-id",
        "H_evidence": "evidence-id",
        "unlock_identity": "unlock-id",
        "best_checkpoint_sha256": "ckpt-sha",
        "config_hash": "config-sha",
        "training_started": True,
        "lse_checkpoint_generated": True,
    }


def _metrics(nll: float = 0.4) -> dict[str, object]:
    return {
        "nll": nll,
        "12": {"n": 8, "nll": 0.7, "mae": 0.4, "rmse": 0.5, "brier": 0.03, "ece": 0.18},
        "18": {"n": 8, "nll": 0.1, "mae": 0.2, "rmse": 0.23, "brier": 0.02, "ece": 0.15},
    }


def test_qualify_lse_accepts_matching_finite_metrics() -> None:
    decision = qual.qualify_lse_evaluation(
        receipt=_receipt(),
        metrics=_metrics(),
        expected={
            "accepted_identity": "accepted-id",
            "v5_deployment_identity": "deploy-id",
            "H_decision": "decision-id",
            "H_evidence": "evidence-id",
            "config_hash": "config-sha",
            "best_checkpoint_sha256": "ckpt-sha",
        },
        max_calibration_nll=0.5,
        required_depths=(12, 18),
    )

    assert decision["verdict"] == "qualified"
    assert decision["accepted_artifact_generated"] is False


def test_qualify_lse_rejects_receipt_identity_mismatch() -> None:
    receipt = _receipt()
    receipt["accepted_identity"] = "wrong"

    with pytest.raises(qual.B2LSEQualificationError) as exc:
        qual.qualify_lse_evaluation(
            receipt=receipt,
            metrics=_metrics(),
            expected={
                "accepted_identity": "accepted-id",
                "v5_deployment_identity": "deploy-id",
                "H_decision": "decision-id",
                "H_evidence": "evidence-id",
                "config_hash": "config-sha",
                "best_checkpoint_sha256": "ckpt-sha",
            },
            max_calibration_nll=0.5,
            required_depths=(12, 18),
        )

    assert exc.value.code == "B2_LSE_QUALIFICATION_IDENTITY_MISMATCH"


def test_qualify_lse_rejects_nll_above_threshold() -> None:
    with pytest.raises(qual.B2LSEQualificationError) as exc:
        qual.qualify_lse_evaluation(
            receipt=_receipt(),
            metrics=_metrics(nll=0.6),
            expected={
                "accepted_identity": "accepted-id",
                "v5_deployment_identity": "deploy-id",
                "H_decision": "decision-id",
                "H_evidence": "evidence-id",
                "config_hash": "config-sha",
                "best_checkpoint_sha256": "ckpt-sha",
            },
            max_calibration_nll=0.5,
            required_depths=(12, 18),
        )

    assert exc.value.code == "B2_LSE_QUALIFICATION_THRESHOLD_FAILED"

from __future__ import annotations

import pytest

from rad.phase_b import b2_phase_final_closure as closure


def _accepted_gate() -> dict[str, object]:
    return {
        "schema_version": "b2_06a_lse_accepted_gate_preflight_evidence_v1",
        "frozen_scientific_identities": {
            "accepted_v5_identity": "dlcm-id",
            "v5_deployment_identity": "deploy-id",
            "beta_star_decimal": "0.54",
        },
        "preflight_result": {
            "accepted_gate_passed": True,
            "training_started": False,
        },
        "stopped_boundaries": {
            "lse_training_started": False,
            "lse_checkpoint_generated": False,
            "push_performed": False,
            "pull_request_opened": False,
        },
    }


def _reference_packaging() -> dict[str, object]:
    return {
        "schema_version": "b2_06b_accepted_v5_reference_packaging_evidence_v1",
        "frozen_identities": {
            "accepted_identity": "dlcm-id",
            "v5_deployment_identity": "deploy-id",
            "beta_star_decimal": "0.54",
        },
        "checkpoint_sha256": "dlcm-checkpoint-sha",
        "boundary": {
            "accepted_identity_changed": False,
            "final_re_evaluated": False,
            "lse_training_started": False,
            "lse_checkpoint_generated": False,
            "push_performed": False,
        },
    }


def _prerequisites() -> dict[str, object]:
    return {
        "schema_version": "b2_06c_lse_prerequisite_materialization_evidence_v1",
        "accepted_gate": {
            "ready": True,
            "accepted_gate_passed": True,
            "training_started": False,
            "accepted_identity": "dlcm-id",
            "v5_deployment_identity": "deploy-id",
        },
        "boundaries": {
            "lse_training_started": False,
            "lse_checkpoint_generated": False,
            "accepted_v5_artifact_unchanged": True,
            "pushed": False,
            "pr_opened": False,
        },
    }


def _training() -> dict[str, object]:
    return {
        "schema_version": "b2_06d_lse_first_controlled_run_evidence_v1",
        "training_git_sha": "training-git",
        "unlock": {
            "unlock_identity": "unlock-id",
            "config_sha256": "config-sha",
        },
        "accepted_gate": {
            "accepted_gate_passed": True,
            "ready": True,
            "accepted_identity": "dlcm-id",
            "v5_deployment_identity": "deploy-id",
        },
        "run": {
            "seed": 111,
            "selector_signal_layout_hash": "selector-hash",
        },
        "artifacts": {
            "best_checkpoint": {
                "sha256": "lse-checkpoint-sha",
                "tracked": False,
            },
            "training_receipt": {
                "receipt_identity": "training-receipt-id",
            },
        },
        "boundaries": {
            "final_content_accessed": False,
            "lse_checkpoint_tracked": False,
            "push": False,
            "pr": False,
        },
    }


def _qualification() -> dict[str, object]:
    return {
        "schema_version": "b2_06e_lse_qualification_decision_v1",
        "verdict": "qualified",
        "H_lse_qualification": "qualification-id",
        "accepted_artifact_generated": False,
        "accepted_identity": "dlcm-id",
        "v5_deployment_identity": "deploy-id",
        "unlock_identity": "unlock-id",
        "training_receipt_identity": "training-receipt-id",
        "best_checkpoint_sha256": "lse-checkpoint-sha",
        "calibration_nll": 0.4,
        "max_calibration_nll": 0.5,
        "evaluated_rows": 16,
    }


def _accepted_lse_manifest() -> dict[str, object]:
    return {
        "schema_version": "b2_06f_lse_accepted_artifact_manifest_v1",
        "lse_qualified": True,
        "accepted_artifact_generated": True,
        "training_started": False,
        "evaluation_started": False,
        "accepted_lse_identity": "lse-id",
        "accepted_dlcm_identity": "dlcm-id",
        "v5_deployment_identity": "deploy-id",
        "unlock_identity": "unlock-id",
        "training_receipt_identity": "training-receipt-id",
        "H_lse_qualification": "qualification-id",
        "lse_checkpoint_sha256": "lse-checkpoint-sha",
        "accepted_lse_checkpoint_sha256": "lse-checkpoint-sha",
        "selector_signal_layout_hash": "selector-hash",
    }


def _accepted_lse_receipt() -> dict[str, object]:
    return {
        "schema_version": "b2_06f_lse_accepted_artifact_closure_receipt_v1",
        "accepted_lse_identity": "lse-id",
        "accepted_lse_checkpoint_sha256": "lse-checkpoint-sha",
        "receipt_identity": "lse-receipt-id",
        "training_started": False,
        "evaluation_started": False,
        "accepted_artifact_generated": True,
    }


def _accepted_lse_evidence() -> dict[str, object]:
    return {
        "schema_version": "b2_06f_accepted_lse_artifact_closure_evidence_v1",
        "status": "accepted_lse_artifact_frozen_locally",
        "accepted_lse_identity": "lse-id",
        "closure_receipt_identity": "lse-receipt-id",
        "accepted_lse_checkpoint_sha256": "lse-checkpoint-sha",
        "selector_signal_layout_hash": "selector-hash",
        "upstream": {
            "H_lse_qualification": "qualification-id",
            "accepted_dlcm_identity": "dlcm-id",
            "v5_deployment_identity": "deploy-id",
            "unlock_identity": "unlock-id",
            "training_receipt_identity": "training-receipt-id",
        },
        "boundary": {
            "training_started_in_06f": False,
            "evaluation_started_in_06f": False,
            "final_content_accessed_in_06f": False,
            "lse_checkpoint_generated_in_06f": False,
            "tracked_pt_files": 0,
            "pushed": False,
            "pr_opened": False,
        },
    }


def _build(**overrides: object) -> dict[str, object]:
    payload = {
        "accepted_gate_evidence": _accepted_gate(),
        "reference_packaging_evidence": _reference_packaging(),
        "prerequisite_evidence": _prerequisites(),
        "training_evidence": _training(),
        "qualification_decision": _qualification(),
        "accepted_lse_manifest": _accepted_lse_manifest(),
        "accepted_lse_receipt": _accepted_lse_receipt(),
        "accepted_lse_evidence": _accepted_lse_evidence(),
        "git_sha": "closure-git",
        "tracked_pt_count": 0,
        "pushed": False,
        "pr_opened": False,
    }
    payload.update(overrides)
    return closure.build_phase_final_closure_manifest(**payload)


def test_build_phase_final_closure_manifest_binds_complete_lse_chain() -> None:
    manifest = _build()

    assert manifest["schema_version"] == "b2_07_phase_final_closure_manifest_v1"
    assert manifest["status"] == "b2_phase_completed_locally"
    assert manifest["accepted_lse_identity"] == "lse-id"
    assert manifest["accepted_dlcm_identity"] == "dlcm-id"
    assert manifest["v5_deployment_identity"] == "deploy-id"
    assert manifest["tracked_pt_count"] == 0
    assert manifest["training_started_in_b2_07"] is False
    assert manifest["evaluation_started_in_b2_07"] is False


def test_build_phase_final_closure_rejects_unqualified_lse() -> None:
    decision = _qualification()
    decision["verdict"] = "failed"

    with pytest.raises(closure.B2PhaseFinalClosureError) as exc:
        _build(qualification_decision=decision)

    assert exc.value.code == "B2_PHASE_FINAL_CLOSURE_LSE_NOT_QUALIFIED"


def test_build_phase_final_closure_rejects_accepted_lse_identity_mismatch() -> None:
    receipt = _accepted_lse_receipt()
    receipt["accepted_lse_identity"] = "other-lse-id"

    with pytest.raises(closure.B2PhaseFinalClosureError) as exc:
        _build(accepted_lse_receipt=receipt)

    assert exc.value.code == "B2_PHASE_FINAL_CLOSURE_IDENTITY_MISMATCH"


def test_build_phase_final_closure_rejects_pushed_boundary() -> None:
    evidence = _accepted_lse_evidence()
    evidence["boundary"] = {**evidence["boundary"], "pushed": True}  # type: ignore[index]

    with pytest.raises(closure.B2PhaseFinalClosureError) as exc:
        _build(accepted_lse_evidence=evidence)

    assert exc.value.code == "B2_PHASE_FINAL_CLOSURE_BOUNDARY_VIOLATION"


def test_build_phase_final_closure_rejects_tracked_pt_files() -> None:
    with pytest.raises(closure.B2PhaseFinalClosureError) as exc:
        _build(tracked_pt_count=1)

    assert exc.value.code == "B2_PHASE_FINAL_CLOSURE_TRACKED_PT"

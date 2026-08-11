from __future__ import annotations

import pytest

from rad.phase_b import b4_dlcm_weight_evidence as evidence

ACCEPTED_DLCM = "0c1a411317f212e5deb29040d184d57aead8a6f862fe3146937db99d1f365116"
V5_DEPLOYMENT = "c56248c9ff6021fc16cf4792d87afeebf1bb8f6d45859f7c26017830dcf0e0bd"
CKPT_SHA = "12b9192643d457eb07745391b68cfa5afe48ec6165b28091bdabde29ec3ece4f"


def _accepted_evidence() -> dict[str, object]:
    return {
        "schema_version": "b2_06b_accepted_v5_reference_packaging_evidence_v1",
        "checkpoint_sha256": CKPT_SHA,
        "frozen_identities": {
            "accepted_identity": ACCEPTED_DLCM,
            "v5_deployment_identity": V5_DEPLOYMENT,
            "beta_star_decimal": "0.54",
        },
        "boundary": {
            "accepted_identity_changed": False,
            "accepted_manifest_modified": False,
            "final_re_evaluated": False,
            "lse_training_started": False,
            "lse_checkpoint_generated": False,
            "push_performed": False,
        },
    }


def _rows() -> list[dict[str, object]]:
    return [
        {
            "stable_sample_id": "a",
            "category": "bottle",
            "split": "calibration",
            "depth": 24,
            "player_layer_ids": [6, 12, 18, 24],
            "dynamic_weights": [0.10, 0.20, 0.30, 0.40],
            "deployment_weights": [0.181, 0.227, 0.273, 0.319],
        },
        {
            "stable_sample_id": "b",
            "category": "carpet",
            "split": "calibration",
            "depth": 24,
            "player_layer_ids": [6, 12, 18, 24],
            "dynamic_weights": [0.40, 0.30, 0.20, 0.10],
            "deployment_weights": [0.319, 0.273, 0.227, 0.181],
        },
    ]


def test_weight_evidence_reports_non_uniform_sample_adaptive_variation() -> None:
    result = evidence.build_weight_evidence_manifest(
        rows=_rows(),
        accepted_reference_evidence=_accepted_evidence(),
        tracked_pt_count=0,
    )

    assert result["schema_version"] == "b4_01_dlcm_adaptive_weight_evidence_manifest_v1"
    assert result["status"] == "dlcm_adaptive_weight_evidence_frozen"
    assert result["accepted_dlcm_identity"] == ACCEPTED_DLCM
    assert result["v5_deployment_identity"] == V5_DEPLOYMENT
    assert result["calibration_records"] == 2
    assert result["uniform_equivalent_at_tolerance"] is False
    assert result["sample_adaptive_variation_observed"] is True
    assert result["dynamic_weight_summary"]["max_sample_linf_delta_from_uniform"] > 0.0
    assert result["deployment_weight_summary"]["max_layer_std"] > 0.0
    assert result["boundary"]["final_content_accessed"] is False


def test_uniform_rows_do_not_overclaim_adaptive_variation() -> None:
    rows = _rows()
    for row in rows:
        row["dynamic_weights"] = [0.25, 0.25, 0.25, 0.25]
        row["deployment_weights"] = [0.25, 0.25, 0.25, 0.25]

    result = evidence.build_weight_evidence_manifest(
        rows=rows,
        accepted_reference_evidence=_accepted_evidence(),
        tracked_pt_count=0,
    )

    assert result["uniform_equivalent_at_tolerance"] is True
    assert result["sample_adaptive_variation_observed"] is False


def test_identity_mismatch_fails_closed() -> None:
    accepted = _accepted_evidence()
    accepted["frozen_identities"] = dict(accepted["frozen_identities"])
    accepted["frozen_identities"]["accepted_identity"] = "wrong"

    with pytest.raises(evidence.B4DLCMWeightEvidenceError) as exc:
        evidence.build_weight_evidence_manifest(
            rows=_rows(),
            accepted_reference_evidence=accepted,
            tracked_pt_count=0,
        )

    assert exc.value.code == "B4_DLCM_WEIGHT_EVIDENCE_IDENTITY_MISMATCH"


def test_tracked_pt_files_fail_closed() -> None:
    with pytest.raises(evidence.B4DLCMWeightEvidenceError) as exc:
        evidence.build_weight_evidence_manifest(
            rows=_rows(),
            accepted_reference_evidence=_accepted_evidence(),
            tracked_pt_count=1,
        )

    assert exc.value.code == "B4_DLCM_WEIGHT_EVIDENCE_TRACKED_PT"

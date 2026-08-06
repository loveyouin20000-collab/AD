from __future__ import annotations

import pytest

from rad.phase_b import b4_final_paper_release as release

ACCEPTED_DLCM = "0c1a411317f212e5deb29040d184d57aead8a6f862fe3146937db99d1f365116"
V5_DEPLOYMENT = "c56248c9ff6021fc16cf4792d87afeebf1bb8f6d45859f7c26017830dcf0e0bd"
ACCEPTED_LSE = "3dafdde6309599d7e82ca6da07db4efbdb09f16105262351c890c514277f01fa"
B2_CLOSURE = "2b1e74c13bba260a9f62c4167b322ae067ecce34fc86a92ae66e1a71b0f3073d"
B3_CLOSURE = "a984814c1821dbc6c0b2ee49fbf018be0c8b4f2fe226855f6b3e015eb89e05be"
B4_WEIGHT = "68bcea45e1fe98ffbee9f9ea51a2b645916b4a623198f787ce8830b1b0f8fe79"


def _b2() -> dict[str, object]:
    return {
        "schema_version": "b2_08_paper_results_evidence_index_manifest_v1",
        "status": "paper_results_and_evidence_index_frozen_locally",
        "source_phase_final_closure_identity": B2_CLOSURE,
        "primary_identities": {
            "v5_deployment_identity": V5_DEPLOYMENT,
            "accepted_dlcm_identity": ACCEPTED_DLCM,
            "accepted_lse_identity": ACCEPTED_LSE,
        },
        "artifact_hashes": {
            "accepted_v5_checkpoint_sha256": "v5sha",
            "accepted_lse_checkpoint_sha256": "lsesha",
        },
        "dlcm": {
            "candidate_layers": [6, 12, 18, 24],
            "accepted_variant": "uniform_anchored_v5",
            "beta_star_decimal": "0.54",
        },
        "lse_qualification": {"verdict": "qualified", "calibration_nll": 0.47},
        "evidence_documents": ["docs/phase_b/b2_08_paper_results_summary.md"],
        "boundary": {
            "training_started_in_b2_08": False,
            "evaluation_started_in_b2_08": False,
            "final_content_accessed_in_b2_08": False,
            "model_artifact_generated_in_b2_08": False,
            "tracked_pt_files": 0,
            "pushed": False,
            "pr_opened": False,
        },
    }


def _b3() -> dict[str, object]:
    return {
        "schema_version": "b3_06_early_exit_phase_closure_manifest_v1",
        "status": "early_exit_phase_closed_negative_result",
        "paper_position": "negative_result_and_future_work",
        "accepted_system_behavior": "full_depth_fallback",
        "phase_closure_identity": B3_CLOSURE,
        "primary_identities": {
            "accepted_dlcm_identity": ACCEPTED_DLCM,
            "accepted_lse_identity": ACCEPTED_LSE,
            "b2_phase_final_closure_identity": B2_CLOSURE,
            "v5_deployment_identity": V5_DEPLOYMENT,
        },
        "claims": {
            "dynamic_fusion_abandoned": False,
            "lse_abandoned": False,
            "early_exit_accepted_mechanism": False,
            "full_depth_fallback_retained": True,
        },
        "evidence_documents": ["docs/phase_b/b3_06_early_exit_paper_results_summary.md"],
        "boundary": {
            "training_started_in_b3_06": False,
            "evaluation_started_in_b3_06": False,
            "final_content_accessed_in_b3_06": False,
            "model_artifact_generated_in_b3_06": False,
            "tracked_pt_files": 0,
            "pushed": False,
            "pr_opened": False,
        },
    }


def _b4() -> dict[str, object]:
    return {
        "schema_version": "b4_01_dlcm_adaptive_weight_evidence_manifest_v1",
        "status": "dlcm_adaptive_weight_evidence_frozen",
        "accepted_dlcm_identity": ACCEPTED_DLCM,
        "v5_deployment_identity": V5_DEPLOYMENT,
        "weight_evidence_identity": B4_WEIGHT,
        "sample_adaptive_variation_observed": True,
        "uniform_equivalent_at_tolerance": False,
        "deployment_weight_summary": {"max_sample_linf_delta_from_uniform": 0.157},
        "boundary": {
            "training_started": False,
            "evaluation_started": False,
            "final_content_accessed": False,
            "model_artifact_generated": False,
            "tracked_pt_files": 0,
            "pushed": False,
            "pr_opened": False,
        },
    }


def test_final_release_binds_paper_claims_without_starting_new_work() -> None:
    result = release.build_final_paper_release_manifest(
        b2_manifest=_b2(),
        b3_manifest=_b3(),
        b4_weight_manifest=_b4(),
        tracked_pt_count=0,
    )

    assert result["schema_version"] == "b4_02_final_local_paper_release_manifest_v1"
    assert result["status"] == "final_local_paper_release_frozen"
    assert result["release_decision"] == "ready_for_push_pr_decision"
    assert result["primary_claims"]["dlcm_sample_adaptive_fusion_supported"] is True
    assert result["primary_claims"]["lse_qualified"] is True
    assert result["primary_claims"]["early_exit_negative_result"] is True
    assert result["primary_claims"]["early_exit_accepted_mechanism"] is False
    assert result["bound_identities"]["b4_01_weight_evidence_identity"] == B4_WEIGHT
    assert result["boundary"]["training_started_in_release"] is False


def test_missing_adaptive_weight_evidence_fails_closed() -> None:
    b4 = _b4()
    b4["sample_adaptive_variation_observed"] = False

    with pytest.raises(release.B4FinalPaperReleaseError) as exc:
        release.build_final_paper_release_manifest(
            b2_manifest=_b2(),
            b3_manifest=_b3(),
            b4_weight_manifest=b4,
            tracked_pt_count=0,
        )

    assert exc.value.code == "B4_FINAL_RELEASE_WEIGHT_EVIDENCE_INVALID"


def test_early_exit_accepted_claim_fails_closed() -> None:
    b3 = _b3()
    b3["claims"] = dict(b3["claims"])
    b3["claims"]["early_exit_accepted_mechanism"] = True

    with pytest.raises(release.B4FinalPaperReleaseError) as exc:
        release.build_final_paper_release_manifest(
            b2_manifest=_b2(),
            b3_manifest=b3,
            b4_weight_manifest=_b4(),
            tracked_pt_count=0,
        )

    assert exc.value.code == "B4_FINAL_RELEASE_EARLY_EXIT_CLAIM_INVALID"


def test_identity_mismatch_fails_closed() -> None:
    b4 = _b4()
    b4["accepted_dlcm_identity"] = "wrong"

    with pytest.raises(release.B4FinalPaperReleaseError) as exc:
        release.build_final_paper_release_manifest(
            b2_manifest=_b2(),
            b3_manifest=_b3(),
            b4_weight_manifest=b4,
            tracked_pt_count=0,
        )

    assert exc.value.code == "B4_FINAL_RELEASE_IDENTITY_MISMATCH"


def test_tracked_pt_files_fail_closed() -> None:
    with pytest.raises(release.B4FinalPaperReleaseError) as exc:
        release.build_final_paper_release_manifest(
            b2_manifest=_b2(),
            b3_manifest=_b3(),
            b4_weight_manifest=_b4(),
            tracked_pt_count=1,
        )

    assert exc.value.code == "B4_FINAL_RELEASE_TRACKED_PT"

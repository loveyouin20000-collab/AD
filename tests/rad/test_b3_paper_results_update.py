from __future__ import annotations

import pytest

from rad.phase_b import b3_paper_results_update as update


ACCEPTED_DLCM = "0c1a411317f212e5deb29040d184d57aead8a6f862fe3146937db99d1f365116"
V5_DEPLOYMENT = "c56248c9ff6021fc16cf4792d87afeebf1bb8f6d45859f7c26017830dcf0e0bd"
ACCEPTED_LSE = "3dafdde6309599d7e82ca6da07db4efbdb09f16105262351c890c514277f01fa"
B2_CLOSURE = "2b1e74c13bba260a9f62c4167b322ae067ecce34fc86a92ae66e1a71b0f3073d"
B3_CLOSURE = "a984814c1821dbc6c0b2ee49fbf018be0c8b4f2fe226855f6b3e015eb89e05be"
B4_WEIGHT = "68bcea45e1fe98ffbee9f9ea51a2b645916b4a623198f787ce8830b1b0f8fe79"
B4_RELEASE = "296191577c12aa42e2e4dbad3d34deaef67b04bbd34d3d0f52be20b9e1c99b93"


def _b2() -> dict[str, object]:
    return {
        "schema_version": "b2_08_paper_results_evidence_index_manifest_v1",
        "source_phase_final_closure_identity": B2_CLOSURE,
        "primary_identities": {
            "accepted_dlcm_identity": ACCEPTED_DLCM,
            "v5_deployment_identity": V5_DEPLOYMENT,
            "accepted_lse_identity": ACCEPTED_LSE,
        },
        "dlcm": {"beta_star_decimal": "0.54"},
        "lse_qualification": {"verdict": "qualified", "calibration_nll": 0.4768362585455179},
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
        "phase_closure_identity": B3_CLOSURE,
        "primary_identities": {
            "accepted_dlcm_identity": ACCEPTED_DLCM,
            "v5_deployment_identity": V5_DEPLOYMENT,
            "accepted_lse_identity": ACCEPTED_LSE,
            "b2_phase_final_closure_identity": B2_CLOSURE,
        },
        "claims": {
            "dynamic_fusion_abandoned": False,
            "lse_abandoned": False,
            "early_exit_accepted_mechanism": False,
            "full_depth_fallback_retained": True,
        },
        "negative_result_table": [
            {
                "candidate_depths": [12, 18],
                "fallback_depth": 24,
                "positive_signal_count": 0,
                "positive_exit_targets": 0,
                "accepted_as_final_mechanism": False,
            }
        ],
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


def _b4_weight() -> dict[str, object]:
    return {
        "schema_version": "b4_01_dlcm_adaptive_weight_evidence_manifest_v1",
        "accepted_dlcm_identity": ACCEPTED_DLCM,
        "v5_deployment_identity": V5_DEPLOYMENT,
        "weight_evidence_identity": B4_WEIGHT,
        "calibration_records": 8,
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


def _b4_release() -> dict[str, object]:
    return {
        "schema_version": "b4_02_final_local_paper_release_manifest_v1",
        "final_release_identity": B4_RELEASE,
        "bound_identities": {
            "accepted_dlcm_identity": ACCEPTED_DLCM,
            "v5_deployment_identity": V5_DEPLOYMENT,
            "accepted_lse_identity": ACCEPTED_LSE,
            "b2_phase_final_closure_identity": B2_CLOSURE,
            "b3_06_phase_closure_identity": B3_CLOSURE,
            "b4_01_weight_evidence_identity": B4_WEIGHT,
        },
        "primary_claims": {
            "dlcm_sample_adaptive_fusion_supported": True,
            "dlcm_final_accepted": True,
            "lse_qualified": True,
            "early_exit_negative_result": True,
            "early_exit_accepted_mechanism": False,
            "full_depth_fallback_retained": True,
        },
        "boundary": {
            "training_started_in_release": False,
            "evaluation_started_in_release": False,
            "final_content_accessed_in_release": False,
            "model_artifact_generated_in_release": False,
            "tracked_pt_files": 0,
            "pushed": False,
            "pr_opened": False,
        },
    }


def test_builds_a_versioned_update_from_frozen_evidence() -> None:
    result = update.build_b3_paper_results_update_manifest(
        b2_manifest=_b2(),
        b3_manifest=_b3(),
        b4_weight_manifest=_b4_weight(),
        b4_release_manifest=_b4_release(),
        tracked_pt_count=0,
    )

    assert result["schema_version"] == "b3_07_paper_results_update_manifest_v1"
    assert result["status"] == "paper_results_update_frozen_locally"
    assert result["paper_claims"]["dlcm_sample_adaptive_fusion_supported"] is True
    assert result["paper_claims"]["lse_qualified"] is True
    assert result["paper_claims"]["early_exit_accepted_mechanism"] is False
    assert result["bound_identities"]["b4_02_final_release_identity"] == B4_RELEASE
    assert result["boundary"]["training_started"] is False
    assert result["boundary"]["final_content_accessed"] is False


def test_wrong_final_release_identity_fails_closed() -> None:
    b4_release = _b4_release()
    b4_release["final_release_identity"] = "wrong"

    with pytest.raises(update.B3PaperResultsUpdateError) as exc:
        update.build_b3_paper_results_update_manifest(
            b2_manifest=_b2(),
            b3_manifest=_b3(),
            b4_weight_manifest=_b4_weight(),
            b4_release_manifest=b4_release,
            tracked_pt_count=0,
        )

    assert exc.value.code == "B3_PAPER_RESULTS_UPDATE_IDENTITY_MISMATCH"


def test_accepted_early_exit_claim_fails_closed() -> None:
    b3 = _b3()
    b3["claims"] = dict(b3["claims"])
    b3["claims"]["early_exit_accepted_mechanism"] = True

    with pytest.raises(update.B3PaperResultsUpdateError) as exc:
        update.build_b3_paper_results_update_manifest(
            b2_manifest=_b2(),
            b3_manifest=b3,
            b4_weight_manifest=_b4_weight(),
            b4_release_manifest=_b4_release(),
            tracked_pt_count=0,
        )

    assert exc.value.code == "B3_PAPER_RESULTS_UPDATE_EARLY_EXIT_CLAIM_INVALID"


def test_tracked_pt_files_fail_closed() -> None:
    with pytest.raises(update.B3PaperResultsUpdateError) as exc:
        update.build_b3_paper_results_update_manifest(
            b2_manifest=_b2(),
            b3_manifest=_b3(),
            b4_weight_manifest=_b4_weight(),
            b4_release_manifest=_b4_release(),
            tracked_pt_count=1,
        )

    assert exc.value.code == "B3_PAPER_RESULTS_UPDATE_TRACKED_PT"


def test_positive_early_exit_signal_fails_closed() -> None:
    b3 = _b3()
    b3["negative_result_table"] = [dict(b3["negative_result_table"][0])]
    b3["negative_result_table"][0]["positive_signal_count"] = 1

    with pytest.raises(update.B3PaperResultsUpdateError) as exc:
        update.build_b3_paper_results_update_manifest(
            b2_manifest=_b2(),
            b3_manifest=b3,
            b4_weight_manifest=_b4_weight(),
            b4_release_manifest=_b4_release(),
            tracked_pt_count=0,
        )

    assert exc.value.code == "B3_PAPER_RESULTS_UPDATE_EARLY_EXIT_RESULT_INVALID"

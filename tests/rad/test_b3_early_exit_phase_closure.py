from __future__ import annotations

import pytest

from rad.phase_b import b3_early_exit_phase_closure as closure

ACCEPTED_DLCM = "0c1a411317f212e5deb29040d184d57aead8a6f862fe3146937db99d1f365116"
V5_DEPLOYMENT = "c56248c9ff6021fc16cf4792d87afeebf1bb8f6d45859f7c26017830dcf0e0bd"
ACCEPTED_LSE = "3dafdde6309599d7e82ca6da07db4efbdb09f16105262351c890c514277f01fa"
LSE_CKPT = "e6e5a4dbd7471ef9e52430eab9533f8edda57ca76ead2ffbed034044805b1c98"
B2_CLOSURE = "2b1e74c13bba260a9f62c4167b322ae067ecce34fc86a92ae66e1a71b0f3073d"
B3_02 = "d77c2cd604c7d3eca4cc9f0649bc8e6ef1b84985a31ec470dd6dd7ef1f43e5b8"
B3_03 = "073afd7f5afeddf1f039034b01bd1ddea84deb14b272b7573d603f978fd2f7b9"
B3_04 = "ef98d0ba9114fd4deb4bccfd98bfacf45a3caddbe0e193e09a31aa8dffd8ab4b"
B3_05 = "f281a3bda75d723a45f8942934c7c4d131e3424d63ba65125d7b6d2cb4ad7cb1"


def _b3_01() -> dict[str, object]:
    return {
        "schema_version": "b3_01_early_exit_contract_preflight_evidence_v1",
        "status": "early_exit_gate_wired_fail_closed",
        "accepted_chain": {
            "accepted_dlcm_identity": ACCEPTED_DLCM,
            "v5_deployment_identity": V5_DEPLOYMENT,
            "accepted_lse_identity": ACCEPTED_LSE,
            "accepted_lse_checkpoint_sha256": LSE_CKPT,
            "b2_phase_final_closure_identity": B2_CLOSURE,
        },
        "preflight_result": {
            "training_started": False,
            "evaluation_started": False,
            "final_content_accessed": False,
            "artifact_written": False,
            "early_depths": [12, 18],
            "full_depth": 24,
        },
        "boundary": {"tracked_pt_files": 0, "pushed": False, "pr_opened": False},
    }


def _b3_02() -> dict[str, object]:
    return {
        "schema_version": "b3_02_exit_prerequisite_materialization_evidence_v1",
        "status": "exit_prerequisites_materialized_locally",
        "accepted_chain": {
            "accepted_dlcm_identity": ACCEPTED_DLCM,
            "accepted_lse_identity": ACCEPTED_LSE,
            "accepted_lse_checkpoint_sha256": LSE_CKPT,
            "b2_phase_final_closure_identity": B2_CLOSURE,
        },
        "materialization": {
            "materialization_identity": B3_02,
            "records": 16,
            "counts_by_depth": {"12": 8, "18": 8},
            "target_exits_by_depth": {"12": 0, "18": 0},
        },
        "boundary": {
            "exit_policy_training_started": False,
            "exit_policy_evaluation_started": False,
            "final_content_accessed": False,
            "checkpoint_generated": False,
            "tracked_pt_files": 0,
            "pushed": False,
            "pr_opened": False,
        },
    }


def _b3_03() -> dict[str, object]:
    return {
        "schema_version": "b3_03_exit_policy_training_contract_evidence_v1",
        "status": "conservative_full_depth_fallback",
        "reason": "no_positive_exit_targets",
        "training_contract_identity": B3_03,
        "positive_exit_targets": 0,
        "training_unlocked": False,
        "training_started": False,
        "evaluation_started": False,
        "checkpoint_generated": False,
        "tracked_pt_count": 0,
    }


def _b3_04() -> dict[str, object]:
    return {
        "schema_version": "b3_04_exit_target_positive_signal_contract_evidence_v1",
        "decision": "no_positive_signal_under_conservative_contract",
        "positive_signal_contract_identity": B3_04,
        "positive_signal_count": 0,
        "candidate_positive_counts_by_depth": {"12": 0, "18": 0},
        "training_unlocked": False,
        "training_started": False,
        "evaluation_started": False,
        "checkpoint_generated": False,
        "tracked_pt_count": 0,
    }


def _b3_05() -> dict[str, object]:
    return {
        "schema_version": "b3_05_early_exit_negative_result_evidence_v1",
        "status": "early_exit_line_closed_negative_result",
        "decision": "conservative_full_depth_fallback",
        "reason": "no_legal_positive_exit_signal",
        "line_closure_identity": B3_05,
        "target_exits_by_depth": {"12": 0, "18": 0},
        "candidate_positive_counts_by_depth": {"12": 0, "18": 0},
        "positive_exit_targets": 0,
        "positive_signal_count": 0,
        "training_unlocked": False,
        "training_started": False,
        "evaluation_started": False,
        "checkpoint_generated": False,
        "tracked_pt_count": 0,
    }


def test_phase_closure_integrates_early_exit_as_paper_ready_negative_result() -> None:
    result = closure.build_early_exit_phase_closure(
        b3_01_evidence=_b3_01(),
        b3_02_evidence=_b3_02(),
        b3_03_evidence=_b3_03(),
        b3_04_evidence=_b3_04(),
        b3_05_evidence=_b3_05(),
        tracked_pt_count=0,
    )

    assert result["schema_version"] == "b3_06_early_exit_phase_closure_manifest_v1"
    assert result["status"] == "early_exit_phase_closed_negative_result"
    assert result["paper_position"] == "negative_result_and_future_work"
    assert result["accepted_system_behavior"] == "full_depth_fallback"
    assert result["claims"]["dynamic_fusion_abandoned"] is False
    assert result["claims"]["early_exit_accepted_mechanism"] is False
    assert result["phase_identities"]["b3_05_line_closure_identity"] == B3_05
    assert result["negative_result_table"][0]["positive_signal_count"] == 0
    assert result["boundary"]["training_started_in_b3_06"] is False


def test_positive_signal_claim_fails_closed() -> None:
    b3_05 = _b3_05()
    b3_05["positive_signal_count"] = 1

    with pytest.raises(closure.B3EarlyExitPhaseClosureError) as exc:
        closure.build_early_exit_phase_closure(
            b3_01_evidence=_b3_01(),
            b3_02_evidence=_b3_02(),
            b3_03_evidence=_b3_03(),
            b3_04_evidence=_b3_04(),
            b3_05_evidence=b3_05,
            tracked_pt_count=0,
        )

    assert exc.value.code == "B3_EARLY_EXIT_PHASE_CLOSURE_NEGATIVE_RESULT_INVALID"


def test_training_or_checkpoint_boundary_violation_fails_closed() -> None:
    b3_03 = _b3_03()
    b3_03["training_unlocked"] = True

    with pytest.raises(closure.B3EarlyExitPhaseClosureError) as exc:
        closure.build_early_exit_phase_closure(
            b3_01_evidence=_b3_01(),
            b3_02_evidence=_b3_02(),
            b3_03_evidence=b3_03,
            b3_04_evidence=_b3_04(),
            b3_05_evidence=_b3_05(),
            tracked_pt_count=0,
        )

    assert exc.value.code == "B3_EARLY_EXIT_PHASE_CLOSURE_BOUNDARY_VIOLATION"


def test_identity_mismatch_fails_closed() -> None:
    b3_02 = _b3_02()
    b3_02["accepted_chain"] = dict(b3_02["accepted_chain"])
    b3_02["accepted_chain"]["accepted_lse_identity"] = "wrong"

    with pytest.raises(closure.B3EarlyExitPhaseClosureError) as exc:
        closure.build_early_exit_phase_closure(
            b3_01_evidence=_b3_01(),
            b3_02_evidence=b3_02,
            b3_03_evidence=_b3_03(),
            b3_04_evidence=_b3_04(),
            b3_05_evidence=_b3_05(),
            tracked_pt_count=0,
        )

    assert exc.value.code == "B3_EARLY_EXIT_PHASE_CLOSURE_IDENTITY_MISMATCH"


def test_tracked_pt_files_fail_closed() -> None:
    with pytest.raises(closure.B3EarlyExitPhaseClosureError) as exc:
        closure.build_early_exit_phase_closure(
            b3_01_evidence=_b3_01(),
            b3_02_evidence=_b3_02(),
            b3_03_evidence=_b3_03(),
            b3_04_evidence=_b3_04(),
            b3_05_evidence=_b3_05(),
            tracked_pt_count=1,
        )

    assert exc.value.code == "B3_EARLY_EXIT_PHASE_CLOSURE_TRACKED_PT"

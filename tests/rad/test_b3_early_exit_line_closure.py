from __future__ import annotations

import pytest

from rad.phase_b import b3_early_exit_line_closure as closure

ACCEPTED_DLCM = "0c1a411317f212e5deb29040d184d57aead8a6f862fe3146937db99d1f365116"
ACCEPTED_LSE = "3dafdde6309599d7e82ca6da07db4efbdb09f16105262351c890c514277f01fa"
B2_CLOSURE = "2b1e74c13bba260a9f62c4167b322ae067ecce34fc86a92ae66e1a71b0f3073d"
B3_02 = "d77c2cd604c7d3eca4cc9f0649bc8e6ef1b84985a31ec470dd6dd7ef1f43e5b8"
B3_03 = "073afd7f5afeddf1f039034b01bd1ddea84deb14b272b7573d603f978fd2f7b9"
B3_04 = "ef98d0ba9114fd4deb4bccfd98bfacf45a3caddbe0e193e09a31aa8dffd8ab4b"


def _b3_02_manifest() -> dict[str, object]:
    return {
        "schema_version": "b3_02_exit_prerequisite_materialization_manifest_v1",
        "materialization_identity": B3_02,
        "accepted_dlcm_identity": ACCEPTED_DLCM,
        "accepted_lse_identity": ACCEPTED_LSE,
        "b2_phase_final_closure_identity": B2_CLOSURE,
        "records": 16,
        "counts_by_depth": {"12": 8, "18": 8},
        "target_exits_by_depth": {"12": 0, "18": 0},
        "training_started": False,
        "evaluation_started": False,
        "final_content_accessed": False,
        "checkpoint_generated": False,
    }


def _b3_03_contract() -> dict[str, object]:
    return {
        "schema_version": "b3_03_exit_policy_training_contract_v1",
        "decision": "conservative_full_depth_fallback",
        "reason": "no_positive_exit_targets",
        "training_contract_identity": B3_03,
        "training_unlocked": False,
        "training_started": False,
        "evaluation_started": False,
        "final_content_accessed": False,
        "checkpoint_generated": False,
        "fallback_depth": 24,
        "positive_exit_targets": 0,
        "target_exits_by_depth": {"12": 0, "18": 0},
        "accepted_dlcm_identity": ACCEPTED_DLCM,
        "accepted_lse_identity": ACCEPTED_LSE,
        "b2_phase_final_closure_identity": B2_CLOSURE,
        "b3_02_materialization_identity": B3_02,
        "tracked_pt_count": 0,
    }


def _b3_04_contract() -> dict[str, object]:
    return {
        "schema_version": "b3_04_exit_target_positive_signal_contract_v1",
        "decision": "no_positive_signal_under_conservative_contract",
        "positive_signal_contract_identity": B3_04,
        "training_unlocked": False,
        "training_started": False,
        "evaluation_started": False,
        "final_content_accessed": False,
        "checkpoint_generated": False,
        "accepted_lse_identity": ACCEPTED_LSE,
        "b3_02_materialization_identity": B3_02,
        "positive_signal_count": 0,
        "candidate_positive_counts_by_depth": {"12": 0, "18": 0},
        "records": 16,
        "early_depths": [12, 18],
        "full_depth": 24,
        "tracked_pt_count": 0,
    }


def test_no_positive_signal_closes_early_exit_line_as_negative_result() -> None:
    result = closure.build_early_exit_line_closure(
        b3_02_manifest=_b3_02_manifest(),
        b3_03_training_contract=_b3_03_contract(),
        b3_04_positive_signal_contract=_b3_04_contract(),
        tracked_pt_count=0,
    )

    assert result["schema_version"] == "b3_05_early_exit_line_closure_manifest_v1"
    assert result["status"] == "early_exit_line_closed_negative_result"
    assert result["decision"] == "conservative_full_depth_fallback"
    assert result["reason"] == "no_legal_positive_exit_signal"
    assert result["training_unlocked"] is False
    assert result["training_started"] is False
    assert result["evaluation_started"] is False
    assert result["checkpoint_generated"] is False
    assert result["fallback_depth"] == 24
    assert result["b3_02_materialization_identity"] == B3_02
    assert result["b3_03_training_contract_identity"] == B3_03
    assert result["b3_04_positive_signal_contract_identity"] == B3_04


def test_positive_signal_count_fails_closed_in_negative_line_closure() -> None:
    b3_04 = _b3_04_contract()
    b3_04["positive_signal_count"] = 1

    with pytest.raises(closure.B3EarlyExitLineClosureError) as exc:
        closure.build_early_exit_line_closure(
            b3_02_manifest=_b3_02_manifest(),
            b3_03_training_contract=_b3_03_contract(),
            b3_04_positive_signal_contract=b3_04,
            tracked_pt_count=0,
        )

    assert exc.value.code == "B3_EARLY_EXIT_LINE_CLOSURE_POSITIVE_SIGNAL_PRESENT"


def test_training_unlock_or_checkpoint_generation_fails_closed() -> None:
    b3_03 = _b3_03_contract()
    b3_03["training_unlocked"] = True

    with pytest.raises(closure.B3EarlyExitLineClosureError) as exc:
        closure.build_early_exit_line_closure(
            b3_02_manifest=_b3_02_manifest(),
            b3_03_training_contract=b3_03,
            b3_04_positive_signal_contract=_b3_04_contract(),
            tracked_pt_count=0,
        )

    assert exc.value.code == "B3_EARLY_EXIT_LINE_CLOSURE_BOUNDARY_VIOLATION"


def test_identity_mismatch_fails_closed() -> None:
    b3_04 = _b3_04_contract()
    b3_04["b3_02_materialization_identity"] = "wrong"

    with pytest.raises(closure.B3EarlyExitLineClosureError) as exc:
        closure.build_early_exit_line_closure(
            b3_02_manifest=_b3_02_manifest(),
            b3_03_training_contract=_b3_03_contract(),
            b3_04_positive_signal_contract=b3_04,
            tracked_pt_count=0,
        )

    assert exc.value.code == "B3_EARLY_EXIT_LINE_CLOSURE_IDENTITY_MISMATCH"


def test_tracked_pt_files_fail_closed() -> None:
    with pytest.raises(closure.B3EarlyExitLineClosureError) as exc:
        closure.build_early_exit_line_closure(
            b3_02_manifest=_b3_02_manifest(),
            b3_03_training_contract=_b3_03_contract(),
            b3_04_positive_signal_contract=_b3_04_contract(),
            tracked_pt_count=1,
        )

    assert exc.value.code == "B3_EARLY_EXIT_LINE_CLOSURE_TRACKED_PT"

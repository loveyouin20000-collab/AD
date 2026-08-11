from __future__ import annotations

import pytest

from rad.phase_b import b3_exit_target_redefinition as redefine


def _trace() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "a",
            "depth": 12,
            "target_gain": 0.4,
            "pred_mean": 0.3,
            "pred_suf_prob": 0.2,
        },
        {
            "sample_id": "b",
            "depth": 18,
            "target_gain": 0.05,
            "pred_mean": 0.08,
            "pred_suf_prob": 0.9,
        },
    ]


def _latency() -> dict[str, object]:
    return {
        "schema_version": "b3_02_latency_profile_v1",
        "depth_savings_proxy_vs_full": {"12": 0.5, "18": 0.25, "24": 0.0},
    }


def test_redefinition_creates_positive_only_when_gain_low_and_sufficiency_high() -> None:
    result = redefine.build_positive_signal_contract(
        calibration_trace_rows=_trace(),
        latency_profile=_latency(),
        max_predicted_remaining_gain=0.10,
        min_predicted_sufficiency_probability=0.50,
        accepted_lse_identity="lse",
        b3_02_materialization_identity="mat",
        tracked_pt_count=0,
    )

    assert result["schema_version"] == "b3_04_exit_target_positive_signal_contract_v1"
    assert result["positive_signal_count"] == 1
    assert result["training_unlocked"] is False
    assert result["candidate_positive_counts_by_depth"] == {"12": 0, "18": 1}


def test_current_conservative_thresholds_can_remain_no_positive_and_keep_training_locked() -> None:
    rows = [
        {"sample_id": "a", "depth": 12, "pred_mean": 0.3, "pred_suf_prob": 0.2},
        {"sample_id": "a", "depth": 18, "pred_mean": 0.2, "pred_suf_prob": 0.2},
    ]

    result = redefine.build_positive_signal_contract(
        calibration_trace_rows=rows,
        latency_profile=_latency(),
        max_predicted_remaining_gain=0.10,
        min_predicted_sufficiency_probability=0.50,
        accepted_lse_identity="lse",
        b3_02_materialization_identity="mat",
        tracked_pt_count=0,
    )

    assert result["positive_signal_count"] == 0
    assert result["decision"] == "no_positive_signal_under_conservative_contract"
    assert result["training_unlocked"] is False


def test_tracked_pt_files_fail_closed() -> None:
    with pytest.raises(redefine.B3ExitTargetRedefinitionError) as exc:
        redefine.build_positive_signal_contract(
            calibration_trace_rows=_trace(),
            latency_profile=_latency(),
            max_predicted_remaining_gain=0.10,
            min_predicted_sufficiency_probability=0.50,
            accepted_lse_identity="lse",
            b3_02_materialization_identity="mat",
            tracked_pt_count=1,
        )

    assert exc.value.code == "B3_EXIT_TARGET_REDEFINITION_TRACKED_PT"


def test_non_conservative_thresholds_fail_closed() -> None:
    with pytest.raises(redefine.B3ExitTargetRedefinitionError) as exc:
        redefine.build_positive_signal_contract(
            calibration_trace_rows=_trace(),
            latency_profile=_latency(),
            max_predicted_remaining_gain=1.0,
            min_predicted_sufficiency_probability=0.1,
            accepted_lse_identity="lse",
            b3_02_materialization_identity="mat",
            tracked_pt_count=0,
        )

    assert exc.value.code == "B3_EXIT_TARGET_REDEFINITION_THRESHOLD_UNSAFE"

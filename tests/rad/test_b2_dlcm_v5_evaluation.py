"""V5 evaluation gate tests."""

from __future__ import annotations

import pytest

from rad.phase_b import b2_dlcm_v5_evaluation as evaluation


def test_gates_thresholds_match_v4() -> None:
    assert evaluation.GT_MACRO_MARGIN == 1e-5
    assert evaluation.GT_PER_CATEGORY_SLACK == 1e-4
    assert evaluation.LOC_MACRO_PIXEL_AP_MIN == 0.0
    assert evaluation.LOC_MACRO_PIXEL_AUROC_MIN == -1e-4
    assert evaluation.LOC_MACRO_AUPRO_MIN == -1e-4
    assert evaluation.LOC_PER_CATEGORY_FLOOR == -1e-3


def test_carpet_fail_and_c4_termination() -> None:
    result = evaluation.evaluate_development_gates(
        depth24_gt_kl_macro=0.08,
        depth24_uniform_gt_kl_macro=0.16,
        per_category_gt_kl={"bottle": 0.1, "carpet": 0.05},
        per_category_uniform_gt_kl={"bottle": 0.27, "carpet": 0.042},
        delta_pixel_ap_macro=0.01,
        delta_pixel_auroc_macro=0.0,
        delta_aupro_macro=0.0,
        per_category_localization={
            "bottle": {"delta_pixel_ap": 0.0, "delta_pixel_auroc": 0.0, "delta_aupro": 0.0},
            "carpet": {"delta_pixel_ap": 0.0, "delta_pixel_auroc": 0.0, "delta_aupro": 0.0},
        },
    )
    assert result["passed"] is False
    assert "gt_category_kl:carpet" in result["failed_reasons"]
    with pytest.raises(evaluation.B2DLCMV5EvaluationError) as exc:
        evaluation.require_development_go(result)
    assert exc.value.code == "B2_DLCM_DEVELOPMENT_UNQUALIFIED"
    term = evaluation.c4_termination_payload(failed_reasons=result["failed_reasons"])
    assert term["enter_c5"] is False
    assert term["lse_started"] is False
    assert "Carpet" in term["conclusion"]

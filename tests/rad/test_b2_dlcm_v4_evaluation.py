"""V3 evaluation gate tests."""

from __future__ import annotations

from rad.phase_b import b2_dlcm_v4_evaluation as evaluation


def test_gates_thresholds_unchanged() -> None:
    result = evaluation.evaluate_development_gates(
        depth24_gt_kl_macro=0.1,
        depth24_uniform_gt_kl_macro=0.2,
        per_category_gt_kl={"bottle": 0.1, "carpet": 0.1},
        per_category_uniform_gt_kl={"bottle": 0.2, "carpet": 0.2},
        delta_pixel_ap_macro=0.0,
        delta_pixel_auroc_macro=0.0,
        delta_aupro_macro=0.0,
        per_category_localization={
            "bottle": {"delta_pixel_ap": 0.0, "delta_pixel_auroc": 0.0, "delta_aupro": 0.0},
            "carpet": {"delta_pixel_ap": 0.0, "delta_pixel_auroc": 0.0, "delta_aupro": 0.0},
        },
    )
    assert result["passed"] is True
    assert result["thresholds"]["gt_macro_margin"] == 1e-5
    assert result["thresholds"]["gt_per_category_slack"] == 1e-4


def test_c1_comparison_is_non_blocking() -> None:
    diag = evaluation.c1_comparison_diagnostic(
        v4_metrics={"gt_macro": 0.2},
        c1_metrics={"gt_macro": 0.1},
    )
    assert diag["qualification_blocking"] is False
    assert diag["must_beat_c1"] is False

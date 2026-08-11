"""V5 eligibility and selection tests."""

from __future__ import annotations

import pytest

from rad.phase_b import b2_dlcm_v5 as v5
from rad.phase_b import b2_dlcm_v5_calibration as calibration
from tests.rad.b2_dlcm_v5_fixtures import make_calibration_records, make_ineligible_records


def test_no_eligible_fail_closed() -> None:
    records = make_ineligible_records()
    candidates = [
        calibration.evaluate_beta_candidate(records, beta_index=i) for i in range(v5.BETA_GRID_SIZE)
    ]
    assert all(c["eligible"] is False for c in candidates)
    with pytest.raises(calibration.B2DLCMV5CalibrationError) as exc:
        calibration.select_beta_star(candidates)
    assert exc.value.code == "B2_DLCM_V5_NO_ELIGIBLE_BETA"


def test_selection_larger_beta_tiebreak() -> None:
    # Construct synthetic candidates with identical M_LOO within eps.
    base = {
        "eligible": True,
        "m_loo": 0.01,
        "macro_gt_kl": 0.05,
        "per_category_gt_kl": {"bottle": 0.05, "carpet": 0.05},
        "uniform_macro_gt_kl": 0.1,
        "per_category_uniform_gt_kl": {"bottle": 0.1, "carpet": 0.1},
        "loo": {},
    }
    candidates = []
    for i in range(v5.BETA_GRID_SIZE):
        row = dict(base)
        row["beta_index"] = i
        row["beta"] = i / 100.0
        row["beta_decimal"] = f"{i / 100.0:.2f}"
        row["eligible"] = i in {10, 20, 30}
        if i == 10:
            row["m_loo"] = 0.010000
        elif i == 20:
            row["m_loo"] = 0.010004  # within 1e-5 of 0.01
        elif i == 30:
            row["m_loo"] = 0.010004
            row["macro_gt_kl"] = 0.04
        else:
            row["eligible"] = False
            row["m_loo"] = 0.5
        candidates.append(row)
    selected = calibration.select_beta_star(candidates)
    # Among 10 (m=0.01), 20 (0.010004), 30 (0.010004, better macro):
    # 10 has lower m_loo by >? 0.000004 < 1e-5 so all three are within eps of the best.
    # Best group starts at lowest m (~0.01). Larger beta among {10,20,30} within eps of best.
    # 30 has largest beta among those within eps of 0.01.
    assert selected["beta_index"] == 30


def test_run_calibration_selects_eligible() -> None:
    records = make_calibration_records()
    manifest = calibration.run_calibration(records, process_label="fixture")
    assert len(manifest["candidates"]) == 101
    assert manifest["selected"]["beta_index"] in range(101)
    assert any(c["eligible"] for c in manifest["candidates"])
    winner = next(
        c for c in manifest["candidates"] if c["beta_index"] == manifest["selected"]["beta_index"]
    )
    assert winner["eligible"] is True


def test_teacher_diagnostics_ignored() -> None:
    records = make_calibration_records()
    a = calibration.run_calibration(
        records, process_label="fixture", teacher_diagnostics={"noise": 1e9}
    )
    b = calibration.run_calibration(records, process_label="fixture", teacher_diagnostics=None)
    assert a["selected"] == b["selected"]
    assert a["scientific_identity"] == b["scientific_identity"]
    assert a["teacher_diagnostics_considered_for_selection"] is False

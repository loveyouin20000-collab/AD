"""V5 Calibration A/B equality tests."""

from __future__ import annotations

from rad.phase_b import b2_dlcm_v5_calibration as calibration
from rad.phase_b import b2_dlcm_v5_protocol as protocol
from tests.rad.b2_dlcm_v5_fixtures import make_calibration_records


def test_calibration_ab_byte_equal() -> None:
    records_a = make_calibration_records()
    records_b = make_calibration_records()
    # Fresh independent runs (no shared mutable state).
    man_a = calibration.run_calibration(records_a, process_label="A")
    man_b = calibration.run_calibration(records_b, process_label="B")
    calibration.assert_calibration_ab_equal(man_a, man_b)
    assert man_a["scientific_identity"] == man_b["scientific_identity"]
    assert man_a["selected"]["beta_index"] == man_b["selected"]["beta_index"]
    bytes_a = protocol.canonical_json_bytes(
        {"candidates": man_a["candidates"], "selected": man_a["selected"]}
    )
    bytes_b = protocol.canonical_json_bytes(
        {"candidates": man_b["candidates"], "selected": man_b["selected"]}
    )
    assert bytes_a == bytes_b

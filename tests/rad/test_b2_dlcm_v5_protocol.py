"""V5 protocol tests."""

from __future__ import annotations

import pytest

from rad.phase_b import b2_dlcm_v5_protocol as protocol


def test_error_codes_complete() -> None:
    required = {
        "B2_DLCM_V5_CONTRACT_MISMATCH",
        "B2_DLCM_V5_TRAINING_FORBIDDEN",
        "B2_DLCM_V5_BETA_GRID_INVALID",
        "B2_DLCM_V5_CALIBRATION_INPUT_INVALID",
        "B2_DLCM_V5_NO_ELIGIBLE_BETA",
        "B2_DLCM_V5_CALIBRATION_MISMATCH",
        "B2_DLCM_V5_BETA_SELECTION_INVALID",
        "B2_DLCM_V5_ROSTER_ADOPTION_MISMATCH",
        "B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN",
        "B2_DLCM_DEVELOPMENT_UNQUALIFIED",
        "B2_DLCM_FINAL_MATERIALIZATION_MISMATCH",
        "B2_DLCM_FINAL_EVALUATION_MISMATCH",
        "B2_DLCM_ACCEPTED_MANIFEST_FORBIDDEN",
    }
    assert required.issubset(set(protocol.ERROR_CODES))


def test_forbid_training_and_final() -> None:
    with pytest.raises(protocol.B2DLCMV5ProtocolError) as exc:
        protocol.forbid_training(context="test")
    assert exc.value.code == "B2_DLCM_V5_TRAINING_FORBIDDEN"
    with pytest.raises(protocol.B2DLCMV5ProtocolError) as exc2:
        protocol.forbid_final_content_access(unlocked=False, context="test")
    assert exc2.value.code == "B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN"
    with pytest.raises(protocol.B2DLCMV5ProtocolError) as exc3:
        protocol.forbid_accepted_manifest(allowed=False, context="test")
    assert exc3.value.code == "B2_DLCM_ACCEPTED_MANIFEST_FORBIDDEN"


def test_reject_bypass_flags() -> None:
    with pytest.raises(protocol.B2DLCMV5ProtocolError):
        protocol.reject_bypass_flags({"force_unlock": True})

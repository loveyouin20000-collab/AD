"""V3 protocol tests."""

from __future__ import annotations

import pytest

from rad.phase_b import b2_dlcm_v4_protocol as protocol


def test_error_codes_complete() -> None:
    required = {
        "B2_DLCM_V4_REAL_TRAINING_NOT_ENABLED",
        "B2_DLCM_V4_CONTRACT_MISMATCH",
        "B2_DLCM_CATEGORY_BATCH_INVALID",
        "B2_DLCM_CATEGORY_COVERAGE_INVALID",
        "B2_DLCM_UNIFORM_BASELINE_INVALID",
        "B2_DLCM_RELATIVE_REGRET_INVALID",
        "B2_DLCM_RELATIVE_SMOOTHMAX_INVALID",
        "B2_DLCM_NO_ELIGIBLE_CHECKPOINT",
        "B2_DLCM_ROSTER_ADOPTION_MISMATCH",
        "B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN",
        "B2_DLCM_DEVELOPMENT_UNQUALIFIED",
        "B2_DLCM_FINAL_MATERIALIZATION_MISMATCH",
        "B2_DLCM_FINAL_EVALUATION_MISMATCH",
        "B2_DLCM_ACCEPTED_MANIFEST_FORBIDDEN",
    }
    assert required.issubset(set(protocol.ERROR_CODES))


def test_bypass_flags_rejected() -> None:
    with pytest.raises(protocol.B2DLCMV4ProtocolError) as exc:
        protocol.reject_bypass_flags({"force_unlock": True})
    assert exc.value.code == "B2_DLCM_V4_CONTRACT_MISMATCH"


def test_final_content_forbidden_without_unlock() -> None:
    with pytest.raises(protocol.B2DLCMV4ProtocolError) as exc:
        protocol.forbid_final_content_access(unlocked=False, context="test")
    assert exc.value.code == "B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN"


def test_real_training_gate() -> None:
    protocol.require_real_training_enabled({"real_training_enabled": False}, dry_run=True)
    with pytest.raises(protocol.B2DLCMV4ProtocolError) as exc:
        protocol.require_real_training_enabled({"real_training_enabled": False}, dry_run=False)
    assert exc.value.code == "B2_DLCM_V4_REAL_TRAINING_NOT_ENABLED"

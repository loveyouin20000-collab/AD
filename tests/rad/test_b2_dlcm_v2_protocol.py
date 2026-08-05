"""RED/GREEN tests for B2-05C1 V2 protocol unlocks and identities."""

from __future__ import annotations

import pytest

from rad.phase_b import b2_dlcm_v2_protocol as subject


def test_error_codes_complete() -> None:
    required = {
        "B2_DLCM_V2_REAL_TRAINING_NOT_ENABLED",
        "B2_DLCM_V2_CONTRACT_MISMATCH",
        "B2_DLCM_FINAL_ROSTER_INSUFFICIENT",
        "B2_DLCM_FINAL_ROSTER_OVERLAP",
        "B2_DLCM_FINAL_ROSTER_SOURCE_INVALID",
        "B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN",
        "B2_DLCM_DEVELOPMENT_UNQUALIFIED",
        "B2_DLCM_FINAL_MATERIALIZATION_UNLOCK_REQUIRED",
        "B2_DLCM_FINAL_MATERIALIZATION_UNLOCK_USED",
        "B2_DLCM_FINAL_MATERIALIZATION_MISMATCH",
        "B2_DLCM_FINAL_EVALUATION_UNLOCK_REQUIRED",
        "B2_DLCM_FINAL_EVALUATION_MISMATCH",
        "B2_DLCM_AUXILIARY_DIAGNOSTICS_INVALID",
        "B2_DLCM_FINAL_DECISION_INVALID",
        "B2_DLCM_FINAL_EVIDENCE_INVALID",
        "B2_DLCM_ACCEPTED_MANIFEST_FORBIDDEN",
    }
    assert required.issubset(set(subject.ERROR_CODES))


def test_no_bypass_flags() -> None:
    with pytest.raises(subject.B2DLCMV2ProtocolError, match="B2_DLCM_V2_CONTRACT_MISMATCH"):
        subject.reject_bypass_flags({"force_unlock": True})


def test_real_training_gate() -> None:
    with pytest.raises(subject.B2DLCMV2ProtocolError, match="B2_DLCM_V2_REAL_TRAINING_NOT_ENABLED"):
        subject.require_real_training_enabled({"real_training_enabled": False}, dry_run=False)
    subject.require_real_training_enabled({"real_training_enabled": False}, dry_run=True)


def test_final_content_forbidden_before_unlock() -> None:
    with pytest.raises(
        subject.B2DLCMV2ProtocolError, match="B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN"
    ):
        subject.forbid_final_content_access(unlocked=False, context="test")


def test_materialization_unlock_consume_and_reuse() -> None:
    unlock = subject.build_materialization_unlock(
        development_go=True,
        development_evidence_sha256="a" * 64,
        implementation_commit="deadbeef",
    )
    consumed = subject.consume_materialization_unlock(unlock)
    assert consumed["consumed"] is True
    with pytest.raises(
        subject.B2DLCMV2ProtocolError, match="B2_DLCM_FINAL_MATERIALIZATION_UNLOCK_USED"
    ):
        subject.consume_materialization_unlock(consumed)


def test_development_fail_blocks_unlock() -> None:
    with pytest.raises(subject.B2DLCMV2ProtocolError, match="B2_DLCM_DEVELOPMENT_UNQUALIFIED"):
        subject.build_materialization_unlock(
            development_go=False,
            development_evidence_sha256="a" * 64,
            implementation_commit="deadbeef",
        )


def test_ab_equality() -> None:
    a = {"x": 1, "y": [1, 2]}
    b = {"y": [1, 2], "x": 1}
    digest = subject.assert_ab_equality(label="materialization", a=a, b=b)
    assert len(digest) == 64
    with pytest.raises(
        subject.B2DLCMV2ProtocolError, match="B2_DLCM_FINAL_EVALUATION_MISMATCH"
    ):
        subject.assert_ab_equality(label="evaluation", a=a, b={"x": 2})


def test_h_decision_rejects_development_teacher() -> None:
    with pytest.raises(subject.B2DLCMV2ProtocolError, match="B2_DLCM_FINAL_DECISION_INVALID"):
        subject.build_h_decision(
            gt_target_learning={"development_go": True},
            localization={},
            thresholds={},
            verdict="qualified",
        )
    with pytest.raises(subject.B2DLCMV2ProtocolError, match="B2_DLCM_FINAL_DECISION_INVALID"):
        subject.build_h_decision(
            gt_target_learning={"kl": 0.1},
            localization={"teacher_kl": 0.2},
            thresholds={},
            verdict="qualified",
        )


def test_accepted_forbidden_before_final_pass() -> None:
    with pytest.raises(
        subject.B2DLCMV2ProtocolError, match="B2_DLCM_ACCEPTED_MANIFEST_FORBIDDEN"
    ):
        subject.build_h_accepted(
            h_deploy="a" * 64,
            h_decision="b" * 64,
            h_evidence="c" * 64,
            h_selection="d" * 64,
            upstream={},
            v2_contract_sha256="e" * 64,
            final_passed=False,
        )

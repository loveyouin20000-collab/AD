"""Tests for B2-05C1B official plan pinning."""

from __future__ import annotations

import pytest

from rad.phase_b import b2_dlcm_v2_official as subject


def test_plan_sha_agreement_mismatch() -> None:
    with pytest.raises(subject.B2DLCMV2OfficialError, match="B2_DLCM_V2_CONTRACT_MISMATCH"):
        subject.require_plan_sha_agreement(
            config={"expected_accepted_v2_training_plan_sha256": "a" * 64},
            recomputed="b" * 64,
            cli_expected="a" * 64,
        )


def test_plan_sha_cli_config_mismatch() -> None:
    with pytest.raises(subject.B2DLCMV2OfficialError, match="B2_DLCM_V2_CONTRACT_MISMATCH"):
        subject.require_plan_sha_agreement(
            config={"expected_accepted_v2_training_plan_sha256": "a" * 64},
            recomputed="a" * 64,
            cli_expected="c" * 64,
        )


def test_c1a_pins() -> None:
    assert subject.C1A_CONTRACT_TAG == "b2-dlcm-decoupled-contract-v2"
    assert subject.C1A_ROSTER_COMMIT == "e54f2b44eeb962b05cfb7cf74764e55905f1a8f6"
    assert subject.V2_IMPLEMENTATION_COMMIT == "e5434c1aafb9d1a3dd75408ff09163e2c581f081"

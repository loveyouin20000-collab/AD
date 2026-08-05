"""V5 official plan SHA agreement tests."""

from __future__ import annotations

import pytest

from rad.phase_b import b2_dlcm_v5_official as official


def test_require_plan_sha_agreement_mismatch() -> None:
    config = {"expected_accepted_v5_calibration_plan_sha256": "a" * 64}
    with pytest.raises(official.B2DLCMV5OfficialError) as exc:
        official.require_plan_sha_agreement(
            config=config,
            recomputed="b" * 64,
            cli_expected=None,
        )
    assert exc.value.code == "B2_DLCM_V5_CONTRACT_MISMATCH"


def test_require_plan_sha_agreement_cli_mismatch() -> None:
    config: dict = {}
    with pytest.raises(official.B2DLCMV5OfficialError) as exc:
        official.require_plan_sha_agreement(
            config=config,
            recomputed="a" * 64,
            cli_expected="b" * 64,
        )
    assert exc.value.code == "B2_DLCM_V5_CONTRACT_MISMATCH"


def test_c3_h_deploy_constant() -> None:
    assert official.C3_H_DEPLOY == (
        "28896ef8c46b54240e8664c7236de4397defa3e877daa5a709249562f716449d"
    )

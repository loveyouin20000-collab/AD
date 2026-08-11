"""C3B official V4 wiring tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rad.phase_b import b2_dlcm_v4_official as official

REPO = Path(__file__).resolve().parents[2]


def test_c3a_pins() -> None:
    assert official.C3A_CONTRACT_TAG == "b2-dlcm-uniform-relative-contract-v4"
    assert official.C3A_ADOPTION_COMMIT == "3b2237affbe58a3cdc30d49bbdee4d8145a6a192"
    assert official.V4_IMPLEMENTATION_COMMIT == "aee4eec9fe17d1972195de2f70a91ceaadd5a28f"


def test_plan_sha_agreement_mismatch() -> None:
    with pytest.raises(official.B2DLCMV4OfficialError, match="B2_DLCM_V4_CONTRACT_MISMATCH"):
        official.require_plan_sha_agreement(
            config={"expected_accepted_v4_training_plan_sha256": "a" * 64},
            recomputed="b" * 64,
            cli_expected="a" * 64,
        )


def test_plan_sha_cli_config_mismatch() -> None:
    with pytest.raises(official.B2DLCMV4OfficialError, match="B2_DLCM_V4_CONTRACT_MISMATCH"):
        official.require_plan_sha_agreement(
            config={"expected_accepted_v4_training_plan_sha256": "a" * 64},
            recomputed="a" * 64,
            cli_expected="c" * 64,
        )


def test_official_config_enabled_pins() -> None:
    cfg = json.loads(
        (REPO / "configs/phase_b/b2_dlcm_uniform_relative_official_v4.json").read_text(
            encoding="utf-8"
        )
    )
    assert cfg["contract_stage"] == "b2_05c3b"
    assert cfg["real_training_enabled"] is True
    assert cfg["official_training_enabled"] is True
    assert cfg["development_enabled"] is True
    assert cfg["final_materialization_enabled"] is False
    assert cfg["final_evaluation_enabled"] is False
    assert cfg["expected_contract_tag"] == official.C3A_CONTRACT_TAG
    assert cfg["expected_contract_commit"] == official.C3A_ADOPTION_COMMIT
    assert cfg["expected_implementation_commit"] == official.V4_IMPLEMENTATION_COMMIT
    assert cfg["relative_regret_mode"] == "batch_matched_model_minus_uniform"
    assert cfg["subtract_gate_slack"] is False
    assert cfg["clamp_negative_regret"] is False
    assert cfg["absolute_regret"] is False
    assert cfg["gt_deployment_aggregation"] == "uniform_relative_smooth_max"
    assert cfg["checkpoint_selection"] == "constrained_worst_relative_regret"
    assert cfg["smoothmax_tau"] == 0.05


def test_load_frozen_roster_and_adoption() -> None:
    roster, adoption = official.load_frozen_roster_and_adoption(REPO)
    assert roster["roster_scientific_sha256"] == (
        "267b7b527f13f84f76f69576d01b1532005d0bb7eda792d558ce5dcce1278213"
    )
    assert adoption["selection_reused_without_change"] is True
    assert adoption["final_content_resolved"] is False
    assert adoption["implementation_commit"] == official.V4_IMPLEMENTATION_COMMIT


def test_pinned_plan_sha_in_config() -> None:
    cfg = json.loads(
        (REPO / "configs/phase_b/b2_dlcm_uniform_relative_official_v4.json").read_text(
            encoding="utf-8"
        )
    )
    pinned = cfg.get("expected_accepted_v4_training_plan_sha256")
    assert isinstance(pinned, str) and len(pinned) == 64
    assert pinned == "4979c73a28e0aaffd21f2c6408bb37e90fdc64201bcc326f990543fbbee5650f"


def test_require_plan_sha_rejects_wrong_pin() -> None:
    with pytest.raises(official.B2DLCMV4OfficialError, match="B2_DLCM_V4_CONTRACT_MISMATCH"):
        official.require_plan_sha_agreement(
            config={
                "expected_accepted_v4_training_plan_sha256": (
                    "4979c73a28e0aaffd21f2c6408bb37e90fdc64201bcc326f990543fbbee5650f"
                )
            },
            recomputed="0" * 64,
            cli_expected="4979c73a28e0aaffd21f2c6408bb37e90fdc64201bcc326f990543fbbee5650f",
        )

"""Tests for B2-05C2B official plan pinning."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rad.phase_b import b2_dlcm_v3_official as subject
from rad.phase_b import b2_dlcm_v3_training as v3_training


def test_plan_sha_agreement_mismatch() -> None:
    with pytest.raises(subject.B2DLCMV3OfficialError, match="B2_DLCM_V3_CONTRACT_MISMATCH"):
        subject.require_plan_sha_agreement(
            config={"expected_accepted_v3_training_plan_sha256": "a" * 64},
            recomputed="b" * 64,
            cli_expected="a" * 64,
        )


def test_plan_sha_cli_config_mismatch() -> None:
    with pytest.raises(subject.B2DLCMV3OfficialError, match="B2_DLCM_V3_CONTRACT_MISMATCH"):
        subject.require_plan_sha_agreement(
            config={"expected_accepted_v3_training_plan_sha256": "a" * 64},
            recomputed="a" * 64,
            cli_expected="c" * 64,
        )


def test_c2a_pins() -> None:
    assert subject.C2A_CONTRACT_TAG == "b2-dlcm-category-robust-contract-v3"
    assert subject.C2A_ADOPTION_COMMIT == "c9dceb4be5438aa0c745fbaf4e3cf7bcba528e64"
    assert subject.V3_IMPLEMENTATION_COMMIT == "e4793c1dac29cc1a195fba098fe765d30bf66d74"


def test_run_v3_contract_training_one_epoch(tmp_path: Path) -> None:
    rows = v3_training.build_hermetic_v3_records()
    records = []
    for row in rows:
        records.append(
            SimpleNamespace(
                stable_sample_id=str(row["stable_sample_id"]),
                split=str(row["split"]),
                category=str(row["category"]),
                descriptors=row["descriptors"],
                p_gt=row["p_gt"],
                p_t=row["p_t"],
                phi_gt=row["phi_gt"],
                phi_t=row["phi_t"],
            )
        )
    result = v3_training.run_v3_contract_training(
        output_root=tmp_path / "seed_run",
        seed=17,
        records=records,
        maximum_epochs=1,
        patience=50,
        device="cpu",
        batch_size=4,
        allow_existing_output=False,
        mark_real_training_started=False,
    )
    assert result["status"] in {"completed_epoch", "early_stopped"}
    assert result["last_epoch"] == 1
    assert result["best_epoch"] is not None
    assert "worst_category_kl" in result
    assert result["evaluation_unlocked"] is False

"""Category-balanced sampler tests."""

from __future__ import annotations

import pytest

from rad.phase_b import b2_dlcm_v3_training as training


def test_batches_are_two_plus_two_and_full_coverage() -> None:
    records = [r for r in training.build_hermetic_v3_records() if r["split"] == "training"]
    batches, state = training.build_category_balanced_epoch_batches(
        records,
        epoch=3,
        bottle_seed=11,
        carpet_seed=22,
        batch_order_seed=33,
    )
    assert len(batches) == 4
    id_to_cat = {r["stable_sample_id"]: r["category"] for r in records}
    flat = []
    for batch in batches:
        cats = [id_to_cat[i] for i in batch]
        assert cats.count("bottle") == 2
        assert cats.count("carpet") == 2
        flat.extend(batch)
    assert len(flat) == 16
    assert len(set(flat)) == 16
    assert state.sampler_contract_version == training.SAMPLER_CONTRACT_VERSION


def test_resume_reproduces_same_batches() -> None:
    records = [r for r in training.build_hermetic_v3_records() if r["split"] == "training"]
    a = training.reproduce_epoch_batches_from_seeds(
        records, epoch=1, bottle_seed=7, carpet_seed=8, batch_order_seed=9
    )
    b = training.reproduce_epoch_batches_from_seeds(
        records, epoch=1, bottle_seed=7, carpet_seed=8, batch_order_seed=9
    )
    assert a == b
    c = training.reproduce_epoch_batches_from_seeds(
        records, epoch=2, bottle_seed=7, carpet_seed=8, batch_order_seed=9
    )
    assert a != c


def test_missing_category_fails() -> None:
    records = [r for r in training.build_hermetic_v3_records() if r["split"] == "training"]
    only_bottle = [r for r in records if r["category"] == "bottle"]
    with pytest.raises(training.B2DLCMV3TrainingError) as exc:
        training.build_category_balanced_epoch_batches(
            only_bottle,
            epoch=0,
            bottle_seed=1,
            carpet_seed=2,
            batch_order_seed=3,
        )
    assert exc.value.code == "B2_DLCM_CATEGORY_COVERAGE_INVALID"


def test_no_normal_anomalous_balancing_required() -> None:
    # Sampler does not inspect anomaly labels; hermetic records lack that field.
    records = [r for r in training.build_hermetic_v3_records() if r["split"] == "training"]
    assert all("normal_or_anomalous" not in r for r in records)
    batches, _ = training.build_category_balanced_epoch_batches(
        records, epoch=0, bottle_seed=1, carpet_seed=2, batch_order_seed=3
    )
    assert len(batches) == 4

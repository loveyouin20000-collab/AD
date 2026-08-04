"""B2-04A portability, determinism, and leakage-boundary tests.

These tests guard the properties that must hold on any machine: the scientific
plan identity is independent of input ordering and of repeated evaluation, the
training-access boundary rejects calibration and evaluation records, and neither
the domain module, the CLI, nor the tracked configuration can reach a
target-domain dataset, a teacher checkpoint, or a machine-local path.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pytest

import rad.phase_b.b2_contribution_targets as subject
import tests.rad.b2_contribution_target_fixtures as fixtures

REPO_ROOT = Path(__file__).resolve().parents[2]
DOMAIN_PATH = REPO_ROOT / "rad" / "phase_b" / "b2_contribution_targets.py"
CLI_PATH = REPO_ROOT / "tools" / "create_b2_contribution_targets.py"
QUALIFICATION_CLI_PATH = (
    REPO_ROOT / "tools" / "qualify_b2_contribution_target_reproduction.py"
)
HISTORICAL_GATE_C_SHA256 = "b1efaa3494e3fc93f50c9565ab42632a3c4458735746beb6de7125d75276beda"

TARGET_DOMAIN_MARKERS = ("VisA", "visa", "Visa")
MACHINE_LOCAL_MARKERS = ("/root/", "/home/", "/mnt/", "autodl", "C:\\", "/tmp/")
FORBIDDEN_DOMAIN_SYMBOLS = (
    "load_teacher_bundle",
    "build_backbone",
    "torch.backends",
    "torch.set_grad_enabled",
)
# B2-04B resolves accepted source-domain identities inside the production API,
# so the module names the source adapter. Only the caller may supply its root,
# and every root must pass the source-only guard before any directory read.
ACCEPTED_SOURCE_ADAPTER_SYMBOL = "MVTecAdapter"
SOURCE_ROOT_ARGUMENT = "mvtec_root"
SOURCE_ONLY_GUARD = "forbid_target_domain_access"


@pytest.fixture(scope="module")
def target_fixture() -> Any:
    return fixtures.build_contribution_target_fixture()


@pytest.fixture(scope="module")
def config() -> Any:
    return subject.load_contribution_targets_config(fixtures.TRACKED_CONFIG_PATH)


@pytest.fixture(scope="module")
def input_bundle(target_fixture: Any) -> Any:
    return fixtures.fixture_input_bundle(target_fixture)


def test_plan_hash_is_invariant_under_input_order_permutation(
    config: Any, target_fixture: Any, input_bundle: Any
) -> None:
    baseline = subject.dry_run_contribution_targets(config=config, inputs=input_bundle)
    shuffled_samples = list(target_fixture.samples)
    random.Random(20260729).shuffle(shuffled_samples)
    assert [sample.stable_sample_id for sample in shuffled_samples] != [
        sample.stable_sample_id for sample in target_fixture.samples
    ]
    permuted = subject.dry_run_contribution_targets(
        config=config,
        inputs=fixtures.fixture_input_bundle(target_fixture, samples=shuffled_samples),
    )
    assert permuted["contribution_plan_scientific_sha256"] == (
        baseline["contribution_plan_scientific_sha256"]
    )
    for identity in subject.SEVEN_LAYERED_IDENTITY_KEYS:
        assert permuted[identity] == baseline[identity]


def test_dry_run_twice_yields_the_identical_plan(config: Any, input_bundle: Any) -> None:
    first = subject.dry_run_contribution_targets(config=config, inputs=input_bundle)
    second = subject.dry_run_contribution_targets(config=config, inputs=input_bundle)
    assert first == second


def test_dry_run_is_independent_of_the_seed_and_output_directory(
    config: Any, input_bundle: Any, tmp_path: Path
) -> None:
    first = subject.dry_run_contribution_targets(
        config=config, inputs=input_bundle, seed=0, output_dir=tmp_path / "a"
    )
    second = subject.dry_run_contribution_targets(
        config=config, inputs=input_bundle, seed=12345, output_dir=tmp_path / "b"
    )
    assert first["contribution_plan_scientific_sha256"] == (
        second["contribution_plan_scientific_sha256"]
    )
    assert not (tmp_path / "a").exists()
    assert not (tmp_path / "b").exists()


def test_gt_calibration_and_normalization_never_see_non_training_samples(
    config: Any, target_fixture: Any
) -> None:
    collection = subject.run_contribution_target_collection(
        config=config, inputs=fixtures.fixture_input_bundle(target_fixture)
    )
    training_ids = {
        sample.stable_sample_id for sample in target_fixture.by_membership("training")
    }
    non_training_ids = {
        sample.stable_sample_id
        for sample in target_fixture.samples
        if sample.membership != "training"
    }
    calibration_ids = set(
        collection.calibration_artifact["ordered_training_stable_sample_ids"]
    )
    normalization_ids = set(collection.normalization["ordered_training_stable_sample_ids"])
    assert calibration_ids == training_ids
    assert normalization_ids == training_ids
    assert calibration_ids & non_training_ids == set()
    assert normalization_ids & non_training_ids == set()


def test_training_access_rejects_calibration_and_evaluation_records(
    config: Any, target_fixture: Any
) -> None:
    collection = subject.run_contribution_target_collection(
        config=config, inputs=fixtures.fixture_input_bundle(target_fixture)
    )
    records = list(collection.records)
    normalization = collection.normalization
    for membership in ("calibration", "evaluation"):
        foreign = [row for row in records if row["split_membership"] == membership]
        assert foreign
        with pytest.raises(subject.ContributionTargetError) as excinfo:
            subject.load_targets_for_access(
                foreign, access_mode="training_only", normalization=normalization
            )
        assert getattr(excinfo.value, "code", "") == "B2_TARGET_ACCESS_LEAKAGE"
    training = [row for row in records if row["split_membership"] == "training"]
    views = subject.load_targets_for_access(
        training, access_mode="training_only", normalization=normalization
    )
    assert len(views) == 16


def test_domain_module_never_reaches_a_target_domain_checkpoint_or_local_path() -> None:
    source = DOMAIN_PATH.read_text(encoding="utf-8")
    for marker in TARGET_DOMAIN_MARKERS:
        assert marker not in source
    for marker in MACHINE_LOCAL_MARKERS:
        assert marker not in source
    for symbol in FORBIDDEN_DOMAIN_SYMBOLS:
        assert symbol not in source


def test_domain_module_source_access_is_adapter_scoped_and_caller_supplied() -> None:
    source = DOMAIN_PATH.read_text(encoding="utf-8")
    assert ACCEPTED_SOURCE_ADAPTER_SYMBOL in source
    assert SOURCE_ROOT_ARGUMENT in source
    assert SOURCE_ONLY_GUARD in source
    assert "adapters.mvtec import MVTecAdapter" in source


@pytest.mark.parametrize(
    "root",
    ["visa", "datasets/visa", "visa/candle", "some/target/root"],
)
def test_source_root_indexing_refuses_a_target_domain_root(root: str) -> None:
    with pytest.raises(Exception) as excinfo:
        subject._index_mvtec_source_records(Path(root))
    assert getattr(excinfo.value, "code", "") == "B2_CACHE_TARGET_ACCESS_FORBIDDEN"


def test_source_root_indexing_fails_closed_on_a_missing_root(tmp_path: Path) -> None:
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject._index_mvtec_source_records(tmp_path / "absent_source_root")
    assert getattr(excinfo.value, "code", "") == "B2_CONTRIBUTION_SOURCE_ROOT_MISSING"


def test_cli_never_reaches_a_target_domain_or_machine_local_path() -> None:
    source = CLI_PATH.read_text(encoding="utf-8")
    for marker in TARGET_DOMAIN_MARKERS:
        assert marker not in source
    for path in (CLI_PATH, QUALIFICATION_CLI_PATH):
        portable_source = path.read_text(encoding="utf-8")
        for marker in MACHINE_LOCAL_MARKERS:
            assert marker not in portable_source
        for forbidden in ("load_teacher_bundle", "build_backbone", "torch.load"):
            assert forbidden not in portable_source


def test_tracked_config_is_portable_and_keeps_official_materialization_disabled() -> None:
    raw = fixtures.TRACKED_CONFIG_PATH.read_text(encoding="utf-8")
    payload = json.loads(raw)
    for marker in (*TARGET_DOMAIN_MARKERS, *MACHINE_LOCAL_MARKERS):
        assert marker not in raw
    assert payload["official_materialization_enabled"] is False
    assert payload["contract_stage"] == "b2_04a"
    assert payload["candidate_layers"] == [6, 12, 18, 24]
    assert payload["prediction_depths"] == [12, 18, 24]


def test_official_config_is_portable_and_long_contract_fields_remain_canonical() -> None:
    raw = fixtures.OFFICIAL_CONFIG_PATH.read_text(encoding="utf-8")
    loader_source = DOMAIN_PATH.read_text(encoding="utf-8")
    for marker in (*TARGET_DOMAIN_MARKERS, *MACHINE_LOCAL_MARKERS):
        assert marker not in raw
    assert '"expected_contribution_contract_tag"' in raw
    assert '"expected_contribution_contract_commit"' in raw
    assert '"fixture_contract_plan_sha256"' in raw
    assert '"expected_accepted_input_plan_sha256"' in raw
    assert fixtures.FIXTURE_CONTRACT_PLAN_SHA256 in raw
    assert fixtures.ACCEPTED_INPUT_CONTRIBUTION_PLAN_SHA256 in raw
    assert fixtures.SUPERSEDED_RUN_CONTROL_FIXTURE_PLAN_SHA256 not in raw
    for alias in ('"expected_contract_tag"', '"expected_contract_commit"'):
        assert alias not in raw
        assert alias.strip('"') not in loader_source
    assert hashlib.sha256(fixtures.TRACKED_CONFIG_PATH.read_bytes()).hexdigest() == (
        HISTORICAL_GATE_C_SHA256
    )


def test_candidate_layers_stay_configuration_driven(target_fixture: Any, tmp_path: Path) -> None:
    """A three-layer lattice must flow end to end without touching the module."""

    narrow_fixture = fixtures.build_contribution_target_fixture(
        candidate_layers=(6, 12, 18), prediction_depths=(12, 18)
    )
    payload = fixtures.controlled_official_config_payload(
        candidate_layers=[6, 12, 18], prediction_depths=[12, 18]
    )
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.load_contribution_targets_config(
            fixtures.write_config(tmp_path, payload, name="narrow.json")
        )
    assert getattr(excinfo.value, "code", "") == "B2_CONTRIBUTION_CONFIG_DRIFT"

    # The reusable collection path itself is lattice-agnostic: it must accept the
    # narrower lattice when it is supplied explicitly rather than hard-coded.
    collection = subject.run_contribution_target_collection(
        config=subject.load_contribution_targets_config(fixtures.TRACKED_CONFIG_PATH),
        inputs=fixtures.fixture_input_bundle(narrow_fixture),
        candidate_layers=(6, 12, 18),
        prediction_depths=(12, 18),
    )
    assert collection.plan["candidate_layers"] == [6, 12, 18]
    assert collection.plan["prediction_depths"] == [12, 18]
    assert len(collection.records) == 32

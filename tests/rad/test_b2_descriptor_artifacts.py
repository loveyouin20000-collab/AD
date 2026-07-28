"""B2-03A TDD RED: contract tests for `rad.phase_b.b2_descriptor_artifacts`.

This module does not exist yet. Importing it below fails with ``ModuleNotFoundError``,
which is the expected RED signal for this whole file (pytest collection error).

--------------------------------------------------------------------------------
Design notes / assumed contract for the GREEN implementer (not yet fixed in stone,
but every test below depends on these choices being followed consistently):

* ``DescriptorArtifactsConfig`` is a frozen dataclass loaded from
  ``configs/phase_b/b2_descriptor_artifacts_gate_c.json`` with fields mirroring the
  JSON 1:1 (see ``test_config_has_required_fields`` below for the exact field list).

* ``validate_accepted_teacher_cache(*, manifest, config, cache_root, allow_test_fixture)``
  performs, in this order (cheap-to-expensive, fail-closed at each step):
    1. ``manifest["status"] == "passed"``                         -> B2_DESC_CACHE_STATUS_NOT_PASSED
    2. ``manifest["cache_scientific_sha256"] == config.expected_teacher_cache_scientific_sha256``
                                                                    -> B2_DESC_CACHE_SCIENTIFIC_HASH_MISMATCH
    3. ``manifest["sample_coverage_sha256"] == config.expected_sample_coverage_sha256``
                                                                    -> B2_DESC_CACHE_COVERAGE_HASH_MISMATCH
    4. ``len(manifest["samples"]) == sum(config.required_split_counts.values())`` (32)
                                                                    -> B2_DESC_CACHE_RECORD_COUNT_MISMATCH
    5. artifact_kind == "test_fixture" and not allow_test_fixture  -> B2_DESC_CACHE_TEST_FIXTURE_FORBIDDEN
    6. per sample entry: reload the ``.pt``, verify file bytes sha256 matches the
       manifest-claimed ``record_file_sha256``                    -> B2_DESC_CACHE_FILE_HASH_MISMATCH
    7. per sample entry: recompute
       ``rad.phase_b.b2_teacher_cache.record_scientific_sha256`` from the reloaded
       content and compare to the manifest-claimed ``record_scientific_sha256``
       (and the embedded payload hash)                            -> B2_DESC_CACHE_RECORD_HASH_MISMATCH
    8. per sample entry: the persisted record's causal-map lattice must be complete
       for ``config.prediction_depths`` (reuses
       ``rad.phase_b.b2_teacher_cache.reconstruct_persisted_descriptors``)
                                                                    -> B2_DESC_DEPTH_MISSING
    9. per sample entry: ``descriptor_feature_names`` on the record must equal
       ``rad.models.descriptors.LAYER_DESCRIPTOR_FEATURE_NAMES`` exactly (order-sensitive)
                                                                    -> B2_DESC_FEATURE_ORDER_MISMATCH
    10. per sample entry: descriptor contract identity fields (``descriptor_contract_version``,
        ``descriptor_implementation_sha256``, ``extractor_configuration_sha256``) match the
        authoritative ``descriptor_contract()``                    -> B2_DESC_CONTRACT_IDENTITY_MISMATCH
  Returns ``AcceptedTeacherCache`` with ``.manifest`` (the validated raw manifest),
  ``.plan`` (tuple of objects exposing ``.stable_sample_id`` / ``.membership``, one
  per accepted sample, in manifest order), and ``.entries`` (tuple, one per accepted
  sample; length 32).

* ``reconstruct_descriptor_record(...)`` builds one descriptor record purely from a
  teacher-cache scientific record Mapping (does not touch cache-level bookkeeping),
  reusing ``rad.phase_b.b2_teacher_cache.reconstruct_persisted_descriptors`` for the
  actual math. It computes and embeds ``descriptor_record_scientific_sha256`` as the
  hash of ``descriptor_record_scientific_content(record)`` (which excludes the hash
  field itself, mirroring the teacher-cache record/file hash pattern).

* ``valid_layer_mask_by_depth[depth]`` is a plain list[bool] the same length as
  ``descriptor_by_depth[depth].shape[1]`` (all True today; no partial-validity cases
  exist in the current dataset, but the field is configuration-driven, not hard-coded).

* ``compute_training_normalization_statistics`` fails closed unless it receives
  *exactly* ``config.required_split_counts["training"]`` (16) records, each with
  ``split_membership == "training"`` and a unique ``stable_sample_id`` -- it never
  silently filters non-training records out.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

import rad.phase_b.b2_descriptor_artifacts as subject  # RED: module does not exist yet
import rad.phase_b.b2_teacher_cache as cache_mod
from rad.models.descriptors import LAYER_DESCRIPTOR_FEATURE_NAMES
from tests.rad import b2_descriptor_fixtures as fixtures

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "phase_b" / "b2_descriptor_artifacts_gate_c.json"

EXPECTED_PROFILE_SHA256 = "7af8dba39633743da0380fef9710940cded655f68c9efa8f84f5a52aeddb3c8d"
EXPECTED_SPLIT_V2 = "91570da1fed6d7859d407196b10403581832ae0ff677a1ea7657ca76b91471f0"
EXPECTED_CHECKPOINT_SHA256 = "97bd461163efb96e36cddb1c3adf677e4c4fc2daabb2521021689f30e799b4f4"
EXPECTED_TEACHER_CACHE_SCIENTIFIC_SHA256 = (
    "66d23807e868696a9c4a68ad83399d82df3d33e743a97d97eeb98ac60c0b1b0a"
)
EXPECTED_SAMPLE_COVERAGE_SHA256 = "6e538b902795c377f9992258e307e58b5c0ba0f99cbbe6c3853a81947ca3d76c"
EXPECTED_MAIN_TAG = "b2-main-integration-v1"
EXPECTED_MAIN_COMMIT = "51e18ade0231c7488ef582bde1e9694f933e85eb"
FEATURE_NAMES = tuple(LAYER_DESCRIPTOR_FEATURE_NAMES)
DESCRIPTOR_DIM = 18
CANDIDATE_LAYERS = (6, 12, 18, 24)
PREDICTION_DEPTHS = (12, 18, 24)


# --------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------


def test_config_has_required_fields() -> None:
    config = subject.load_descriptor_artifacts_config(CONFIG_PATH)
    assert config.schema_version == 1
    assert config.configuration_id == "b2_descriptor_artifacts_gate_c"
    assert config.expected_main_tag == EXPECTED_MAIN_TAG
    assert config.expected_main_commit == EXPECTED_MAIN_COMMIT
    assert config.expected_teacher_cache_scientific_sha256 == EXPECTED_TEACHER_CACHE_SCIENTIFIC_SHA256
    assert config.expected_sample_coverage_sha256 == EXPECTED_SAMPLE_COVERAGE_SHA256
    assert config.expected_split_scientific_sha256 == EXPECTED_SPLIT_V2
    assert config.expected_checkpoint_sha256 == EXPECTED_CHECKPOINT_SHA256
    assert config.expected_execution_profile_sha256 == EXPECTED_PROFILE_SHA256
    assert config.candidate_layers == CANDIDATE_LAYERS
    assert config.prediction_depths == PREDICTION_DEPTHS
    assert config.descriptor_dimension == DESCRIPTOR_DIM
    assert config.descriptor_contract_version == 1
    assert config.normalization_contract_version == 1
    assert dict(config.required_split_counts) == {
        "training": 16,
        "calibration": 8,
        "evaluation": 8,
    }
    assert config.primary_dtype == "float32"
    assert "candidate_layers_configuration_driven" in config.fail_closed_requirements


def test_config_json_has_no_machine_local_paths() -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    for forbidden in ("/root/", "/home/", "autodl-tmp", "C:\\\\"):
        assert forbidden not in text


def test_config_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_"):
        subject.load_descriptor_artifacts_config(tmp_path / "missing.json")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw.update(candidate_layers=[6, 12, 24]),
        lambda raw: raw.update(prediction_depths=[18, 24]),
        lambda raw: raw.update(descriptor_dimension=16),
        lambda raw: raw.update(expected_split_scientific_sha256="0" * 64),
        lambda raw: raw.update(expected_checkpoint_sha256="0" * 64),
        lambda raw: raw.update(expected_execution_profile_sha256="0" * 64),
        lambda raw: raw.update(required_split_counts={"training": 15, "calibration": 8, "evaluation": 8}),
    ],
)
def test_config_rejects_drifted_fixed_fields(tmp_path: Path, mutate: Any) -> None:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(raw)
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_"):
        subject.load_descriptor_artifacts_config(changed)


# --------------------------------------------------------------------------------
# Shared fixture/config plumbing
# --------------------------------------------------------------------------------


@pytest.fixture
def descriptor_fixture(tmp_path: Path) -> dict[str, Any]:
    return fixtures.build_descriptor_test_fixture(tmp_path)


@pytest.fixture
def descriptor_config(
    tmp_path: Path, descriptor_fixture: dict[str, Any]
) -> subject.DescriptorArtifactsConfig:
    path = fixtures.write_descriptor_config_json(tmp_path, descriptor_fixture)
    return subject.load_descriptor_artifacts_config(path)


def _manifest(descriptor_fixture: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(descriptor_fixture["manifest"])


# --------------------------------------------------------------------------------
# Cache validation: happy path
# --------------------------------------------------------------------------------


def test_validate_accepted_teacher_cache_accepts_valid_test_fixture(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    accepted = subject.validate_accepted_teacher_cache(
        manifest=_manifest(descriptor_fixture),
        config=descriptor_config,
        cache_root=descriptor_fixture["cache_root"],
        allow_test_fixture=True,
    )
    assert len(accepted.entries) == 32
    assert len(accepted.plan) == 32
    assert Counter(row.membership for row in accepted.plan) == {
        "training": 16,
        "calibration": 8,
        "evaluation": 8,
    }
    assert {row.stable_sample_id for row in accepted.plan} == {
        row.stable_sample_id for row in descriptor_fixture["plan"]
    }
    assert accepted.manifest["status"] == "passed"


# --------------------------------------------------------------------------------
# Cache validation: negative paths
# --------------------------------------------------------------------------------


def test_validate_accepted_teacher_cache_rejects_test_fixture_by_default(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    with pytest.raises(
        subject.DescriptorArtifactsError, match="B2_DESC_CACHE_TEST_FIXTURE_FORBIDDEN"
    ):
        subject.validate_accepted_teacher_cache(
            manifest=_manifest(descriptor_fixture),
            config=descriptor_config,
            cache_root=descriptor_fixture["cache_root"],
            allow_test_fixture=False,
        )


def test_validate_accepted_teacher_cache_rejects_status_not_passed(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    manifest = _manifest(descriptor_fixture)
    manifest["status"] = "partial"
    with pytest.raises(
        subject.DescriptorArtifactsError, match="B2_DESC_CACHE_STATUS_NOT_PASSED"
    ):
        subject.validate_accepted_teacher_cache(
            manifest=manifest,
            config=descriptor_config,
            cache_root=descriptor_fixture["cache_root"],
            allow_test_fixture=True,
        )


def test_validate_accepted_teacher_cache_rejects_scientific_hash_drift(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    manifest = _manifest(descriptor_fixture)
    manifest["cache_scientific_sha256"] = "0" * 64
    with pytest.raises(
        subject.DescriptorArtifactsError, match="B2_DESC_CACHE_SCIENTIFIC_HASH_MISMATCH"
    ):
        subject.validate_accepted_teacher_cache(
            manifest=manifest,
            config=descriptor_config,
            cache_root=descriptor_fixture["cache_root"],
            allow_test_fixture=True,
        )


def test_validate_accepted_teacher_cache_rejects_sample_coverage_hash_drift(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    manifest = _manifest(descriptor_fixture)
    manifest["sample_coverage_sha256"] = "1" * 64
    with pytest.raises(
        subject.DescriptorArtifactsError, match="B2_DESC_CACHE_COVERAGE_HASH_MISMATCH"
    ):
        subject.validate_accepted_teacher_cache(
            manifest=manifest,
            config=descriptor_config,
            cache_root=descriptor_fixture["cache_root"],
            allow_test_fixture=True,
        )


@pytest.mark.parametrize("drop_last", [True, False])
def test_validate_accepted_teacher_cache_rejects_wrong_record_count(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
    drop_last: bool,
) -> None:
    manifest = _manifest(descriptor_fixture)
    if drop_last:
        manifest["samples"] = manifest["samples"][:-1]
    else:
        manifest["samples"] = manifest["samples"] + [manifest["samples"][0]]
    with pytest.raises(
        subject.DescriptorArtifactsError, match="B2_DESC_CACHE_RECORD_COUNT_MISMATCH"
    ):
        subject.validate_accepted_teacher_cache(
            manifest=manifest,
            config=descriptor_config,
            cache_root=descriptor_fixture["cache_root"],
            allow_test_fixture=True,
        )


def test_validate_accepted_teacher_cache_rejects_record_hash_drift(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    manifest = _manifest(descriptor_fixture)
    manifest["samples"][0]["record_scientific_sha256"] = "2" * 64
    with pytest.raises(
        subject.DescriptorArtifactsError, match="B2_DESC_CACHE_RECORD_HASH_MISMATCH"
    ):
        subject.validate_accepted_teacher_cache(
            manifest=manifest,
            config=descriptor_config,
            cache_root=descriptor_fixture["cache_root"],
            allow_test_fixture=True,
        )


def test_validate_accepted_teacher_cache_rejects_file_byte_hash_drift(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    manifest = _manifest(descriptor_fixture)
    stable_id = manifest["samples"][0]["stable_sample_id"]
    fixtures.append_bytes_to_sample_file(descriptor_fixture["cache_root"], stable_id)
    with pytest.raises(
        subject.DescriptorArtifactsError, match="B2_DESC_CACHE_FILE_HASH_MISMATCH"
    ):
        subject.validate_accepted_teacher_cache(
            manifest=manifest,
            config=descriptor_config,
            cache_root=descriptor_fixture["cache_root"],
            allow_test_fixture=True,
        )


def test_validate_accepted_teacher_cache_rejects_missing_required_depth(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    manifest = _manifest(descriptor_fixture)
    stable_id = manifest["samples"][0]["stable_sample_id"]

    def drop_one_causal_map(record: dict[str, Any]) -> dict[str, Any]:
        name = next(iter(sorted(record["tensors"])))
        assert name.startswith("causal_map:")
        del record["tensors"][name]
        return record

    fixtures.rewrite_sample_record(descriptor_fixture, manifest, stable_id, drop_one_causal_map)
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_DEPTH_MISSING"):
        subject.validate_accepted_teacher_cache(
            manifest=manifest,
            config=descriptor_config,
            cache_root=descriptor_fixture["cache_root"],
            allow_test_fixture=True,
        )


def test_validate_accepted_teacher_cache_rejects_feature_order_drift(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    manifest = _manifest(descriptor_fixture)
    stable_id = manifest["samples"][0]["stable_sample_id"]

    def reverse_feature_order(record: dict[str, Any]) -> dict[str, Any]:
        record["descriptor_feature_names"] = list(reversed(record["descriptor_feature_names"]))
        return record

    fixtures.rewrite_sample_record(descriptor_fixture, manifest, stable_id, reverse_feature_order)
    with pytest.raises(
        subject.DescriptorArtifactsError, match="B2_DESC_FEATURE_ORDER_MISMATCH"
    ):
        subject.validate_accepted_teacher_cache(
            manifest=manifest,
            config=descriptor_config,
            cache_root=descriptor_fixture["cache_root"],
            allow_test_fixture=True,
        )


def test_validate_accepted_teacher_cache_rejects_descriptor_contract_identity_drift(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    manifest = _manifest(descriptor_fixture)
    stable_id = manifest["samples"][0]["stable_sample_id"]

    def drift_contract(record: dict[str, Any]) -> dict[str, Any]:
        record["descriptor_implementation_sha256"] = "3" * 64
        return record

    fixtures.rewrite_sample_record(descriptor_fixture, manifest, stable_id, drift_contract)
    with pytest.raises(
        subject.DescriptorArtifactsError, match="B2_DESC_CONTRACT_IDENTITY_MISMATCH"
    ):
        subject.validate_accepted_teacher_cache(
            manifest=manifest,
            config=descriptor_config,
            cache_root=descriptor_fixture["cache_root"],
            allow_test_fixture=True,
        )


# --------------------------------------------------------------------------------
# Descriptor reconstruction
# --------------------------------------------------------------------------------


def _source_record(descriptor_fixture: dict[str, Any], index: int = 0) -> dict[str, Any]:
    stable_id = descriptor_fixture["plan"][index].stable_sample_id
    return fixtures.load_sample_record(descriptor_fixture["cache_root"], stable_id)


def _reconstruct_kwargs(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
    index: int = 0,
) -> dict[str, Any]:
    record = _source_record(descriptor_fixture, index)
    return {
        "teacher_scientific_record": record,
        "config": descriptor_config,
        "source_teacher_record_scientific_sha256": cache_mod.record_scientific_sha256(record),
        "source_teacher_record_file_sha256": next(
            entry["record_file_sha256"]
            for entry in descriptor_fixture["manifest"]["samples"]
            if entry["stable_sample_id"] == record["stable_sample_id"]
        ),
        "teacher_cache_scientific_sha256": descriptor_fixture["teacher_cache_scientific_sha256"],
        "descriptor_feature_order": FEATURE_NAMES,
        "descriptor_extractor_config_sha256": descriptor_fixture["descriptor_contract"][
            "extractor_configuration_sha256"
        ],
        "descriptor_extractor_implementation_sha256": descriptor_fixture["descriptor_contract"][
            "descriptor_implementation_sha256"
        ],
    }


def test_reconstruct_descriptor_record_produces_exact_depth_shapes(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    record = subject.reconstruct_descriptor_record(**_reconstruct_kwargs(descriptor_fixture, descriptor_config))
    shapes = {depth: tuple(tensor.shape) for depth, tensor in record["descriptor_by_depth"].items()}
    assert shapes == {12: (1, 2, 18), 18: (1, 3, 18), 24: (1, 4, 18)}
    for tensor in record["descriptor_by_depth"].values():
        assert tensor.dtype == torch.float32
        assert bool(torch.isfinite(tensor).all())


def test_reconstruct_descriptor_record_layer_ordering_and_valid_masks_are_exact(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    record = subject.reconstruct_descriptor_record(**_reconstruct_kwargs(descriptor_fixture, descriptor_config))
    assert tuple(record["candidate_layers"]) == CANDIDATE_LAYERS
    assert tuple(record["prediction_depths"]) == PREDICTION_DEPTHS
    masks = record["valid_layer_mask_by_depth"]
    assert [bool(v) for v in masks[12]] == [True, True]
    assert [bool(v) for v in masks[18]] == [True, True, True]
    assert [bool(v) for v in masks[24]] == [True, True, True, True]


def test_reconstruct_descriptor_record_matches_authoritative_extractor_output(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    kwargs = _reconstruct_kwargs(descriptor_fixture, descriptor_config)
    record = subject.reconstruct_descriptor_record(**kwargs)
    authoritative = cache_mod.reconstruct_persisted_descriptors(kwargs["teacher_scientific_record"])
    for depth in PREDICTION_DEPTHS:
        assert torch.equal(record["descriptor_by_depth"][depth], authoritative[depth])


def test_reconstruct_descriptor_record_includes_required_fields(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    record = subject.reconstruct_descriptor_record(**_reconstruct_kwargs(descriptor_fixture, descriptor_config))
    for field in (
        "schema_version",
        "stable_sample_id",
        "split_membership",
        "category",
        "label",
        "anomaly_type",
        "candidate_layers",
        "prediction_depths",
        "descriptor_contract_version",
        "descriptor_feature_order",
        "descriptor_extractor_config_sha256",
        "descriptor_extractor_implementation_sha256",
        "descriptor_by_depth",
        "valid_layer_mask_by_depth",
        "source_teacher_record_scientific_sha256",
        "source_teacher_record_file_sha256",
        "teacher_cache_scientific_sha256",
        "split_scientific_sha256",
        "checkpoint_sha256",
        "execution_profile_sha256",
        "descriptor_record_scientific_sha256",
    ):
        assert field in record, field
    assert record["split_membership"] == "training"
    subject.validate_descriptor_record(record, config=descriptor_config)


def test_changing_a_cached_map_changes_descriptor_scientific_hash(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    kwargs = _reconstruct_kwargs(descriptor_fixture, descriptor_config)
    baseline = subject.reconstruct_descriptor_record(**kwargs)
    baseline_hash = subject.descriptor_record_scientific_sha256(baseline)
    assert baseline_hash == baseline["descriptor_record_scientific_sha256"]

    mutated_record = copy.deepcopy(kwargs["teacher_scientific_record"])
    name = next(iter(sorted(mutated_record["tensors"])))
    tensor = mutated_record["tensors"][name]["tensor"].clone()
    tensor[0, 0, 0, 0] += 5.0
    mutated_record["tensors"][name] = dict(mutated_record["tensors"][name])
    mutated_record["tensors"][name]["tensor"] = tensor
    mutated_record["tensors"][name]["digest"] = cache_mod.canonical_tensor_digest(
        name, tensor, tuple(mutated_record["tensors"][name]["dimension_semantics"])
    )
    mutated_kwargs = dict(kwargs)
    mutated_kwargs["teacher_scientific_record"] = mutated_record
    mutated = subject.reconstruct_descriptor_record(**mutated_kwargs)
    assert subject.descriptor_record_scientific_sha256(mutated) != baseline_hash


# --------------------------------------------------------------------------------
# Descriptor record validation: NaN/Inf and schema
# --------------------------------------------------------------------------------


def _hand_built_descriptor_record(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
    **overrides: Any,
) -> dict[str, Any]:
    descriptor_by_depth = {
        12: torch.zeros(1, 2, DESCRIPTOR_DIM, dtype=torch.float32),
        18: torch.zeros(1, 3, DESCRIPTOR_DIM, dtype=torch.float32),
        24: torch.zeros(1, 4, DESCRIPTOR_DIM, dtype=torch.float32),
    }
    valid_layer_mask_by_depth = {
        12: [True, True],
        18: [True, True, True],
        24: [True, True, True, True],
    }
    record: dict[str, Any] = {
        "schema_version": 1,
        "stable_sample_id": "a" * 64,
        "split_membership": "training",
        "category": "bottle",
        "label": 1,
        "anomaly_type": "crack",
        "candidate_layers": list(CANDIDATE_LAYERS),
        "prediction_depths": list(PREDICTION_DEPTHS),
        "descriptor_contract_version": descriptor_fixture["descriptor_contract"][
            "descriptor_contract_version"
        ],
        "descriptor_feature_order": list(FEATURE_NAMES),
        "descriptor_extractor_config_sha256": descriptor_fixture["descriptor_contract"][
            "extractor_configuration_sha256"
        ],
        "descriptor_extractor_implementation_sha256": descriptor_fixture["descriptor_contract"][
            "descriptor_implementation_sha256"
        ],
        "descriptor_by_depth": descriptor_by_depth,
        "valid_layer_mask_by_depth": valid_layer_mask_by_depth,
        "source_teacher_record_scientific_sha256": "b" * 64,
        "source_teacher_record_file_sha256": "c" * 64,
        "teacher_cache_scientific_sha256": descriptor_fixture["teacher_cache_scientific_sha256"],
        "split_scientific_sha256": EXPECTED_SPLIT_V2,
        "checkpoint_sha256": descriptor_fixture["checkpoint_sha256"],
        "execution_profile_sha256": EXPECTED_PROFILE_SHA256,
    }
    record.update(overrides)
    record["descriptor_record_scientific_sha256"] = subject.descriptor_record_scientific_sha256(record)
    return record


def test_validate_descriptor_record_accepts_well_formed_record(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    record = _hand_built_descriptor_record(descriptor_fixture, descriptor_config)
    subject.validate_descriptor_record(record, config=descriptor_config)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_validate_descriptor_record_rejects_nonfinite_descriptor(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
    value: float,
) -> None:
    broken = torch.zeros(1, 2, DESCRIPTOR_DIM, dtype=torch.float32)
    broken[0, 0, 0] = value
    record = _hand_built_descriptor_record(
        descriptor_fixture,
        descriptor_config,
        descriptor_by_depth={
            12: broken,
            18: torch.zeros(1, 3, DESCRIPTOR_DIM, dtype=torch.float32),
            24: torch.zeros(1, 4, DESCRIPTOR_DIM, dtype=torch.float32),
        },
    )
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_TENSOR_NONFINITE"):
        subject.validate_descriptor_record(record, config=descriptor_config)


def test_validate_descriptor_record_rejects_wrong_depth_shape(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    record = _hand_built_descriptor_record(
        descriptor_fixture,
        descriptor_config,
        descriptor_by_depth={
            12: torch.zeros(1, 3, DESCRIPTOR_DIM, dtype=torch.float32),  # wrong: should be 2
            18: torch.zeros(1, 3, DESCRIPTOR_DIM, dtype=torch.float32),
            24: torch.zeros(1, 4, DESCRIPTOR_DIM, dtype=torch.float32),
        },
    )
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_"):
        subject.validate_descriptor_record(record, config=descriptor_config)


def test_descriptor_record_scientific_content_excludes_self_hash_and_includes_whitelist(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    record = _hand_built_descriptor_record(descriptor_fixture, descriptor_config)
    content = subject.descriptor_record_scientific_content(record)
    assert "descriptor_record_scientific_sha256" not in content
    for required in (
        "stable_sample_id",
        "split_membership",
        "candidate_layers",
        "prediction_depths",
        "descriptor_feature_order",
        "descriptor_contract_version",
        "descriptor_extractor_config_sha256",
        "descriptor_extractor_implementation_sha256",
        "source_teacher_record_scientific_sha256",
        "split_scientific_sha256",
        "checkpoint_sha256",
        "execution_profile_sha256",
    ):
        assert required in content, required


def test_descriptor_record_scientific_hash_ignores_paths_branches_and_timestamps(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    record = _hand_built_descriptor_record(descriptor_fixture, descriptor_config)
    baseline = subject.descriptor_record_scientific_sha256(record)
    poisoned = dict(record)
    poisoned["absolute_output_path"] = "/tmp/should-not-hash"
    poisoned["git_branch"] = "some-worktree-branch"
    poisoned["worktree_path"] = "/root/autodl-tmp/AD-phase-b2-descriptor-artifacts"
    poisoned["timestamp"] = "2099-01-01T00:00:00Z"
    poisoned["runtime_attestation_sha256"] = "d" * 64
    poisoned["record_file_sha256"] = "e" * 64
    assert subject.descriptor_record_scientific_sha256(poisoned) == baseline


# --------------------------------------------------------------------------------
# Source-only normalization statistics
# --------------------------------------------------------------------------------


def _all_reconstructed_records(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> list[dict[str, Any]]:
    records = []
    for index in range(len(descriptor_fixture["plan"])):
        kwargs = _reconstruct_kwargs(descriptor_fixture, descriptor_config, index)
        records.append(subject.reconstruct_descriptor_record(**kwargs))
    return records


def _training_records(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> list[dict[str, Any]]:
    return [
        record
        for record in _all_reconstructed_records(descriptor_fixture, descriptor_config)
        if record["split_membership"] == "training"
    ]


def test_compute_training_normalization_statistics_happy_path_shape(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    training_records = _training_records(descriptor_fixture, descriptor_config)
    assert len(training_records) == 16
    statistics = subject.compute_training_normalization_statistics(
        training_records, config=descriptor_config
    )
    for depth in PREDICTION_DEPTHS:
        assert depth in statistics["axes"] or depth in statistics
    coverage = subject.training_sample_coverage_sha256(
        sorted(record["stable_sample_id"] for record in training_records)
    )
    assert statistics["training_sample_coverage_sha256"] == coverage


@pytest.mark.parametrize("bad_membership", ["calibration", "evaluation", "target_domain"])
def test_compute_training_normalization_statistics_rejects_non_training_membership(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
    bad_membership: str,
) -> None:
    training_records = _training_records(descriptor_fixture, descriptor_config)
    poisoned = copy.deepcopy(training_records)
    poisoned[0]["split_membership"] = bad_membership
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_NORMALIZATION_"):
        subject.compute_training_normalization_statistics(poisoned, config=descriptor_config)


def test_compute_training_normalization_statistics_rejects_omitted_training_sample(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    training_records = _training_records(descriptor_fixture, descriptor_config)
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_NORMALIZATION_"):
        subject.compute_training_normalization_statistics(
            training_records[:-1], config=descriptor_config
        )


def test_compute_training_normalization_statistics_rejects_unexpected_extra_sample(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    training_records = _training_records(descriptor_fixture, descriptor_config)
    calibration_record = next(
        record
        for record in _all_reconstructed_records(descriptor_fixture, descriptor_config)
        if record["split_membership"] == "calibration"
    )
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_NORMALIZATION_"):
        subject.compute_training_normalization_statistics(
            [*training_records, calibration_record], config=descriptor_config
        )


def test_compute_training_normalization_statistics_rejects_duplicate_sample(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    training_records = _training_records(descriptor_fixture, descriptor_config)
    duplicated = [*training_records[:-1], training_records[0]]
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_NORMALIZATION_"):
        subject.compute_training_normalization_statistics(duplicated, config=descriptor_config)


def test_compute_training_normalization_statistics_deterministic_under_permutation(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    training_records = _training_records(descriptor_fixture, descriptor_config)
    forward = subject.compute_training_normalization_statistics(
        training_records, config=descriptor_config
    )
    reversed_order = subject.compute_training_normalization_statistics(
        list(reversed(training_records)), config=descriptor_config
    )
    assert subject.normalization_statistics_scientific_sha256(
        forward
    ) == subject.normalization_statistics_scientific_sha256(reversed_order)


def test_normalization_statistics_scientific_hash_ignores_formatting_and_paths(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    training_records = _training_records(descriptor_fixture, descriptor_config)
    statistics = subject.compute_training_normalization_statistics(
        training_records, config=descriptor_config
    )
    baseline = subject.normalization_statistics_scientific_sha256(statistics)
    poisoned = dict(statistics)
    poisoned["output_path"] = "/tmp/somewhere-else"
    poisoned["timestamp"] = "2099-01-01T00:00:00Z"
    poisoned["git_branch"] = "other-worktree"
    assert subject.normalization_statistics_scientific_sha256(poisoned) == baseline


def test_normalization_statistics_change_when_one_descriptor_value_changes(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    training_records = _training_records(descriptor_fixture, descriptor_config)
    baseline = subject.compute_training_normalization_statistics(
        training_records, config=descriptor_config
    )
    mutated = copy.deepcopy(training_records)
    mutated[0]["descriptor_by_depth"][12] = (
        mutated[0]["descriptor_by_depth"][12] + 1.0
    )
    changed = subject.compute_training_normalization_statistics(mutated, config=descriptor_config)
    assert subject.normalization_statistics_scientific_sha256(
        changed
    ) != subject.normalization_statistics_scientific_sha256(baseline)


def test_apply_frozen_normalization_applies_identically_to_calibration_and_evaluation(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    all_records = _all_reconstructed_records(descriptor_fixture, descriptor_config)
    training_records = [r for r in all_records if r["split_membership"] == "training"]
    calibration_records = [r for r in all_records if r["split_membership"] == "calibration"]
    evaluation_records = [r for r in all_records if r["split_membership"] == "evaluation"]
    statistics = subject.compute_training_normalization_statistics(
        training_records, config=descriptor_config
    )
    for record in [*calibration_records, *evaluation_records]:
        original_tensors = {
            depth: record["descriptor_by_depth"][depth].detach().clone()
            for depth in PREDICTION_DEPTHS
        }
        original_tensor_ids = {
            depth: id(record["descriptor_by_depth"][depth]) for depth in PREDICTION_DEPTHS
        }
        normalized = subject.apply_frozen_normalization(record, statistics)
        for depth in PREDICTION_DEPTHS:
            # Must not mutate the input record's tensor storage or replace in place.
            assert id(record["descriptor_by_depth"][depth]) == original_tensor_ids[depth]
            assert torch.equal(record["descriptor_by_depth"][depth], original_tensors[depth])
            assert not torch.equal(
                normalized["descriptor_by_depth"][depth], record["descriptor_by_depth"][depth]
            ) or bool(torch.isfinite(normalized["descriptor_by_depth"][depth]).all())
            assert normalized["descriptor_by_depth"][depth].shape == record["descriptor_by_depth"][depth].shape
            assert bool(torch.isfinite(normalized["descriptor_by_depth"][depth]).all())


def test_apply_frozen_normalization_zero_variance_uses_unit_divisor(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    training_records = _training_records(descriptor_fixture, descriptor_config)
    constant_value = 3.5
    constant_records = copy.deepcopy(training_records)
    for record in constant_records:
        for depth in PREDICTION_DEPTHS:
            record["descriptor_by_depth"][depth] = torch.full_like(
                record["descriptor_by_depth"][depth], constant_value
            )
    statistics = subject.compute_training_normalization_statistics(
        constant_records, config=descriptor_config
    )
    probe = copy.deepcopy(constant_records[0])
    normalized = subject.apply_frozen_normalization(probe, statistics)
    for depth in PREDICTION_DEPTHS:
        tensor = normalized["descriptor_by_depth"][depth]
        assert bool(torch.isfinite(tensor).all())
        assert bool(torch.allclose(tensor, torch.zeros_like(tensor)))


def test_training_sample_coverage_sha256_is_order_invariant_and_change_sensitive(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    training_records = _training_records(descriptor_fixture, descriptor_config)
    ids = [record["stable_sample_id"] for record in training_records]
    forward = subject.training_sample_coverage_sha256(ids)
    shuffled = subject.training_sample_coverage_sha256(list(reversed(ids)))
    assert forward == shuffled
    dropped = subject.training_sample_coverage_sha256(ids[:-1])
    assert dropped != forward
    swapped = [*ids[:-1], "f" * 64]
    assert subject.training_sample_coverage_sha256(swapped) != forward


# --------------------------------------------------------------------------------
# Artifact integrity
# --------------------------------------------------------------------------------


def _descriptor_records_for_membership(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
    membership: str,
) -> list[dict[str, Any]]:
    return [
        record
        for record in _all_reconstructed_records(descriptor_fixture, descriptor_config)
        if record["split_membership"] == membership
    ]


def test_write_descriptor_record_atomic_refuses_overwrite(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    record = subject.reconstruct_descriptor_record(**_reconstruct_kwargs(descriptor_fixture, descriptor_config))
    destination = tmp_path / "descriptors" / f"{record['stable_sample_id']}.pt"
    destination.parent.mkdir(parents=True)
    subject.write_descriptor_record_atomic(destination, record)
    before = destination.read_bytes()
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_"):
        subject.write_descriptor_record_atomic(destination, record)
    assert destination.read_bytes() == before


def test_audit_descriptor_artifact_integrity_rejects_orphan_and_extra(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    run_dir = tmp_path / "descriptor_run"
    (run_dir / "descriptors").mkdir(parents=True)
    training_records = _training_records(descriptor_fixture, descriptor_config)
    planned_ids = [record["stable_sample_id"] for record in training_records]
    for record in training_records:
        destination = run_dir / "descriptors" / f"{record['stable_sample_id']}.pt"
        subject.write_descriptor_record_atomic(destination, record)
    manifest = {"status": "passed", "planned_stable_sample_ids": planned_ids}
    subject.audit_descriptor_artifact_integrity(
        run_dir=run_dir, manifest=manifest, planned_ids=planned_ids
    )

    orphan = run_dir / "descriptors" / "orphan.pt"
    orphan.write_bytes(b"orphan")
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_ORPHAN_ARTIFACT"):
        subject.audit_descriptor_artifact_integrity(
            run_dir=run_dir, manifest=manifest, planned_ids=planned_ids
        )
    orphan.unlink()

    extra_manifest = {
        "status": "passed",
        "planned_stable_sample_ids": [*planned_ids, "f" * 64],
    }
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_"):
        subject.audit_descriptor_artifact_integrity(
            run_dir=run_dir, manifest=extra_manifest, planned_ids=extra_manifest["planned_stable_sample_ids"]
        )


def test_audit_descriptor_artifact_integrity_rejects_missing_record(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    run_dir = tmp_path / "descriptor_run"
    (run_dir / "descriptors").mkdir(parents=True)
    training_records = _training_records(descriptor_fixture, descriptor_config)
    planned_ids = [record["stable_sample_id"] for record in training_records]
    for record in training_records[:-1]:
        destination = run_dir / "descriptors" / f"{record['stable_sample_id']}.pt"
        subject.write_descriptor_record_atomic(destination, record)
    manifest = {"status": "passed", "planned_stable_sample_ids": planned_ids}
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_"):
        subject.audit_descriptor_artifact_integrity(
            run_dir=run_dir, manifest=manifest, planned_ids=planned_ids
        )


def test_audit_descriptor_artifact_integrity_rejects_partial_claiming_passed(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "descriptor_run"
    (run_dir / "descriptors").mkdir(parents=True)
    manifest = {"status": "passed", "planned_stable_sample_ids": ["a" * 64]}
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_"):
        subject.audit_descriptor_artifact_integrity(
            run_dir=run_dir, manifest=manifest, planned_ids=["a" * 64]
        )


def test_audit_descriptor_artifact_integrity_rejects_digest_only_resume_record(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    run_dir = tmp_path / "descriptor_run"
    (run_dir / "descriptors").mkdir(parents=True)
    training_records = _training_records(descriptor_fixture, descriptor_config)
    planned_ids = [record["stable_sample_id"] for record in training_records]
    record = training_records[0]
    destination = run_dir / "descriptors" / f"{record['stable_sample_id']}.pt"
    digest_only_payload = {
        "descriptor_record_scientific_sha256": record["descriptor_record_scientific_sha256"],
        "stable_sample_id": record["stable_sample_id"],
        # Intentionally no "descriptor_by_depth" tensor values -- digest-only resume state.
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(digest_only_payload, destination)
    manifest = {"status": "passed", "planned_stable_sample_ids": planned_ids}
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_"):
        subject.audit_descriptor_artifact_integrity(
            run_dir=run_dir, manifest=manifest, planned_ids=planned_ids
        )


def test_normalization_before_complete_training_coverage_is_forbidden(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    run_dir = tmp_path / "descriptor_run"
    (run_dir / "descriptors").mkdir(parents=True)
    training_records = _training_records(descriptor_fixture, descriptor_config)
    planned_ids = [record["stable_sample_id"] for record in training_records]
    # Only 15 of 16 training descriptor artifacts are actually on disk.
    for record in training_records[:-1]:
        destination = run_dir / "descriptors" / f"{record['stable_sample_id']}.pt"
        subject.write_descriptor_record_atomic(destination, record)
    manifest = {
        "status": "passed",
        "planned_stable_sample_ids": planned_ids,
        "normalization_statistics": {"computed": True},
    }
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_"):
        subject.audit_descriptor_artifact_integrity(
            run_dir=run_dir, manifest=manifest, planned_ids=planned_ids
        )


# --------------------------------------------------------------------------------
# Extraction planning / manifest building smoke tests
# --------------------------------------------------------------------------------


def test_build_descriptor_extraction_plan_summarizes_accepted_cache(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    accepted = subject.validate_accepted_teacher_cache(
        manifest=_manifest(descriptor_fixture),
        config=descriptor_config,
        cache_root=descriptor_fixture["cache_root"],
        allow_test_fixture=True,
    )
    plan_summary = subject.build_descriptor_extraction_plan(accepted=accepted, config=descriptor_config)
    assert plan_summary["planned_samples"] == 32
    assert plan_summary["training_samples_for_normalization"] == 16
    assert plan_summary["calibration_samples_for_normalization"] == 0
    assert plan_summary["evaluation_samples_for_normalization"] == 0
    assert list(plan_summary["prediction_depths"]) == list(PREDICTION_DEPTHS)
    assert plan_summary["descriptor_dimension"] == DESCRIPTOR_DIM


def test_build_descriptor_artifacts_manifest_reports_passed_status(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
    tmp_path: Path,
) -> None:
    training_records = _training_records(descriptor_fixture, descriptor_config)
    statistics = subject.compute_training_normalization_statistics(
        training_records, config=descriptor_config
    )
    all_records = _all_reconstructed_records(descriptor_fixture, descriptor_config)
    run_dir = tmp_path / "passed_manifest_run"
    entries = [
        subject.write_descriptor_record_atomic(
            run_dir / subject.descriptor_relative_path(record["stable_sample_id"]),
            record,
        )
        for record in all_records
    ]
    normalization = subject.write_normalization_statistics_atomic(
        run_dir / "normalization_statistics.pt",
        statistics,
    )
    manifest = subject.build_descriptor_artifacts_manifest(
        config=descriptor_config,
        records=all_records,
        statistics=statistics,
        sample_entries=entries,
        normalization_entry=normalization,
    )
    assert manifest["status"] == "passed"
    assert manifest["descriptor_dimension"] == DESCRIPTOR_DIM
    assert "normalization_statistics_scientific_sha256" in manifest
    for key in (
        "descriptor_collection_scientific_sha256",
        "descriptor_sample_coverage_sha256",
        "normalization_training_coverage_sha256",
    ):
        assert key in manifest, key


# --------------------------------------------------------------------------------
# B2-03B review regressions: disk-authoritative accepted cache + fail-closed finalization
# --------------------------------------------------------------------------------


def test_recompute_teacher_cache_hashes_match_authoritative_fixture(
    descriptor_fixture: dict[str, Any],
) -> None:
    manifest = _manifest(descriptor_fixture)
    assert (
        cache_mod.recompute_teacher_cache_sample_coverage_sha256(
            descriptor_fixture["entries"]
        )
        == "6e538b902795c377f9992258e307e58b5c0ba0f99cbbe6c3853a81947ca3d76c"
    )
    assert (
        cache_mod.recompute_teacher_cache_scientific_sha256(
            verified_entries=descriptor_fixture["entries"],
            manifest_contract=manifest,
        )
        == manifest["cache_scientific_sha256"]
    )


def test_validate_accepted_teacher_cache_rejects_manifest_summary_hash_that_disagrees_with_entries(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    manifest = fixtures.production_like_manifest(descriptor_fixture)
    stable_id = manifest["samples"][0]["stable_sample_id"]

    def mutate_tensor(record: dict[str, Any]) -> dict[str, Any]:
        name = next(key for key in sorted(record["tensors"]) if key.startswith("causal_map:"))
        tensor = record["tensors"][name]["tensor"].clone()
        tensor[0, 0, 0, 0] += 1.0
        record["tensors"][name] = dict(record["tensors"][name])
        record["tensors"][name]["tensor"] = tensor
        record["tensors"][name]["digest"] = cache_mod.canonical_tensor_digest(
            name, tensor, tuple(record["tensors"][name]["dimension_semantics"])
        )
        return record

    fixtures.rewrite_sample_record(descriptor_fixture, manifest, stable_id, mutate_tensor)
    manifest["cache_scientific_sha256"] = descriptor_config.expected_teacher_cache_scientific_sha256
    manifest["sample_coverage_sha256"] = descriptor_config.expected_sample_coverage_sha256
    with pytest.raises(
        subject.DescriptorArtifactsError, match="B2_DESC_CACHE_SCIENTIFIC_HASH_MISMATCH"
    ):
        subject.validate_accepted_teacher_cache(
            manifest=manifest,
            config=descriptor_config,
            cache_root=descriptor_fixture["cache_root"],
            allow_test_fixture=True,
        )


def test_load_disk_authoritative_manifest_rejects_manifest_outside_root(
    tmp_path: Path, descriptor_fixture: dict[str, Any]
) -> None:
    outside_root = tmp_path / "outside-root"
    outside_root.mkdir()
    outside_manifest = tmp_path / "teacher_cache_manifest.json"
    outside_manifest.write_text(
        json.dumps(fixtures.production_like_manifest(descriptor_fixture)),
        encoding="utf-8",
    )
    with pytest.raises(
        subject.DescriptorArtifactsError, match="B2_DESC_CACHE_MANIFEST_OUTSIDE_ROOT"
    ):
        subject.load_disk_authoritative_teacher_cache_manifest(
            teacher_cache_manifest_path=outside_manifest,
            teacher_cache_root=outside_root,
        )


def test_validate_accepted_teacher_cache_rejects_record_path_escaping_cache_root(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    manifest = fixtures.production_like_manifest(descriptor_fixture)
    manifest["samples"][0]["relative_path"] = "../escaped.pt"
    with pytest.raises(
        subject.DescriptorArtifactsError, match="B2_DESC_CACHE_PATH_ESCAPE"
    ):
        subject.validate_accepted_teacher_cache(
            manifest=manifest,
            config=descriptor_config,
            cache_root=descriptor_fixture["cache_root"],
            allow_test_fixture=True,
        )


def test_validate_accepted_teacher_cache_rejects_wrong_17_7_8_membership_distribution(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    manifest = fixtures.production_like_manifest(descriptor_fixture)
    stable_id = next(
        row.stable_sample_id
        for row in descriptor_fixture["plan"]
        if row.membership == "calibration"
    )

    def mutate_membership(record: dict[str, Any]) -> dict[str, Any]:
        record["membership"] = "training"
        return record

    fixtures.rewrite_sample_record(descriptor_fixture, manifest, stable_id, mutate_membership)
    with pytest.raises(
        subject.DescriptorArtifactsError, match="B2_DESC_CACHE_SPLIT_COUNT_MISMATCH"
    ):
        subject.validate_accepted_teacher_cache(
            manifest=manifest,
            config=descriptor_config,
            cache_root=descriptor_fixture["cache_root"],
            allow_test_fixture=True,
        )


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("checkpoint_sha256", "1" * 64, "B2_DESC_COLLECTION_PROVENANCE_MISMATCH"),
        ("execution_profile_sha256", "2" * 64, "B2_DESC_COLLECTION_PROVENANCE_MISMATCH"),
        ("descriptor_implementation_sha256", "3" * 64, "B2_DESC_CONTRACT_IDENTITY_MISMATCH"),
    ],
)
def test_validate_accepted_teacher_cache_rejects_per_record_provenance_mismatch(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
    field: str,
    value: str,
    code: str,
) -> None:
    manifest = fixtures.production_like_manifest(descriptor_fixture)
    stable_id = manifest["samples"][0]["stable_sample_id"]

    def mutate(record: dict[str, Any]) -> dict[str, Any]:
        record[field] = value
        return record

    fixtures.rewrite_sample_record(descriptor_fixture, manifest, stable_id, mutate)
    with pytest.raises(subject.DescriptorArtifactsError, match=code):
        subject.validate_accepted_teacher_cache(
            manifest=manifest,
            config=descriptor_config,
            cache_root=descriptor_fixture["cache_root"],
            allow_test_fixture=True,
        )


def test_validate_accepted_teacher_cache_rejects_config_checkpoint_profile_or_split_mismatch(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    drifted = replace(
        descriptor_config,
        expected_checkpoint_sha256="4" * 64,
        expected_execution_profile_sha256="5" * 64,
        expected_split_scientific_sha256="6" * 64,
    )
    with pytest.raises(
        subject.DescriptorArtifactsError, match="B2_DESC_COLLECTION_PROVENANCE_MISMATCH"
    ):
        subject.validate_accepted_teacher_cache(
            manifest=fixtures.production_like_manifest(descriptor_fixture),
            config=drifted,
            cache_root=descriptor_fixture["cache_root"],
            allow_test_fixture=False,
        )


def test_write_descriptor_record_atomic_reloads_and_revalidates_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    record = subject.reconstruct_descriptor_record(
        **_reconstruct_kwargs(descriptor_fixture, descriptor_config)
    )
    destination = tmp_path / "descriptors" / f"{record['stable_sample_id']}.pt"
    original_torch_load = subject.torch.load
    observed = {"calls": 0}

    def corrupted_load(*args: Any, **kwargs: Any) -> Any:
        payload = original_torch_load(*args, **kwargs)
        observed["calls"] += 1
        if observed["calls"] == 1:
            mutated = copy.deepcopy(payload)
            mutated["scientific_record"]["descriptor_by_depth"][12] = (
                mutated["scientific_record"]["descriptor_by_depth"][12] + 1.0
            )
            return mutated
        return payload

    monkeypatch.setattr(subject.torch, "load", corrupted_load)
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_RECORD_HASH_MISMATCH"):
        subject.write_descriptor_record_atomic(destination, record)


def test_write_normalization_statistics_atomic_reloads_and_revalidates_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    statistics = subject.compute_training_normalization_statistics(
        _training_records(descriptor_fixture, descriptor_config),
        config=descriptor_config,
    )
    destination = tmp_path / "normalization_statistics.pt"
    original_torch_load = subject.torch.load
    observed = {"calls": 0}

    def corrupted_load(*args: Any, **kwargs: Any) -> Any:
        payload = original_torch_load(*args, **kwargs)
        observed["calls"] += 1
        if observed["calls"] == 1:
            mutated = copy.deepcopy(payload)
            mutated["scientific_statistics_record"]["axes"][12]["layers"][0]["features"][0][
                "mean"
            ] += 1.0
            return mutated
        return payload

    monkeypatch.setattr(subject.torch, "load", corrupted_load)
    with pytest.raises(
        subject.DescriptorArtifactsError, match="B2_DESC_NORMALIZATION_HASH_MISMATCH"
    ):
        subject.write_normalization_statistics_atomic(destination, statistics)


def test_build_descriptor_artifacts_manifest_requires_persisted_sample_entries_for_passed_status(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    statistics = subject.compute_training_normalization_statistics(
        _training_records(descriptor_fixture, descriptor_config),
        config=descriptor_config,
    )
    with pytest.raises(
        subject.DescriptorArtifactsError, match="B2_DESC_PASSED_MANIFEST_REQUIRES_VERIFIED_DISK"
    ):
        subject.build_descriptor_artifacts_manifest(
            config=descriptor_config,
            records=_all_reconstructed_records(descriptor_fixture, descriptor_config),
            statistics=statistics,
        )


@pytest.mark.parametrize(
    "mutate_entry",
    [
        lambda entry: subject.PersistedDescriptorEntry(
            stable_sample_id=entry.stable_sample_id,
            relative_record_path=entry.relative_record_path,
            descriptor_record_scientific_sha256=entry.descriptor_record_scientific_sha256,
            descriptor_record_file_sha256="",
            verification_status=entry.verification_status,
        ),
        lambda entry: subject.PersistedDescriptorEntry(
            stable_sample_id=entry.stable_sample_id,
            relative_record_path=entry.relative_record_path,
            descriptor_record_scientific_sha256=entry.descriptor_record_scientific_sha256,
            descriptor_record_file_sha256=entry.descriptor_record_file_sha256,
            verification_status="planned",
        ),
    ],
)
def test_build_descriptor_artifacts_manifest_rejects_incomplete_verified_entries(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
    mutate_entry: Any,
) -> None:
    all_records = _all_reconstructed_records(descriptor_fixture, descriptor_config)
    run_dir = tmp_path / "run"
    entries = []
    for record in all_records:
        entries.append(
            subject.write_descriptor_record_atomic(
                run_dir / subject.descriptor_relative_path(record["stable_sample_id"]),
                record,
            )
        )
    bad_entries = list(entries)
    bad_entries[0] = mutate_entry(entries[0])
    statistics = subject.compute_training_normalization_statistics(
        [row for row in all_records if row["split_membership"] == "training"],
        config=descriptor_config,
    )
    normalization = subject.write_normalization_statistics_atomic(
        run_dir / "normalization_statistics.pt",
        statistics,
    )
    with pytest.raises(
        subject.DescriptorArtifactsError, match="B2_DESC_PASSED_MANIFEST_REQUIRES_VERIFIED_DISK"
    ):
        subject.build_descriptor_artifacts_manifest(
            config=descriptor_config,
            records=all_records,
            statistics=statistics,
            sample_entries=bad_entries,
            normalization_entry=normalization,
        )


def test_planned_manifest_cannot_be_confused_with_passed_manifest(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    accepted = subject.validate_accepted_teacher_cache(
        manifest=fixtures.production_like_manifest(descriptor_fixture),
        config=descriptor_config,
        cache_root=descriptor_fixture["cache_root"],
        allow_test_fixture=True,
    )
    planned = subject.build_planned_descriptor_artifacts_manifest(
        accepted=accepted,
        config=descriptor_config,
    )
    assert planned["status"] == "planned"
    assert "samples" not in planned
    assert "normalization_statistics_file_sha256" not in planned


def test_verify_descriptor_artifact_collection_requires_passed_manifest_config_and_run_root(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    run_dir = tmp_path / "descriptor_run"
    run_dir.mkdir()
    manifest = {"status": "passed"}
    (run_dir / "final_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "final_manifest.json.sha256").write_text("0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_MANIFEST_RECEIPT_MISMATCH"):
        subject.verify_descriptor_artifact_collection(
            config=descriptor_config,
            run_dir=run_dir,
        )


def test_materialize_descriptor_artifact_collection_fixture_dual_run_is_scientifically_stable(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    manifest_path = descriptor_fixture["cache_root"] / "teacher_cache_manifest.json"
    manifest_path.write_text(
        json.dumps(fixtures.production_like_manifest(descriptor_fixture), sort_keys=True),
        encoding="utf-8",
    )
    first = subject.materialize_descriptor_artifact_collection(
        config=descriptor_config,
        teacher_cache_manifest_path=manifest_path,
        teacher_cache_root=descriptor_fixture["cache_root"],
        output_run_dir=tmp_path / "run-one",
    )
    second = subject.materialize_descriptor_artifact_collection(
        config=descriptor_config,
        teacher_cache_manifest_path=manifest_path,
        teacher_cache_root=descriptor_fixture["cache_root"],
        output_run_dir=tmp_path / "run-two",
    )
    assert first.teacher_forward_count == 0
    assert second.teacher_forward_count == 0
    assert len(list((tmp_path / "run-one" / "descriptors").glob("*.pt"))) == 32
    assert len(list((tmp_path / "run-two" / "descriptors").glob("*.pt"))) == 32
    first_verified = subject.verify_descriptor_artifact_collection(
        config=descriptor_config,
        run_dir=tmp_path / "run-one",
    )
    second_verified = subject.verify_descriptor_artifact_collection(
        config=descriptor_config,
        run_dir=tmp_path / "run-two",
    )
    comparison = subject.compare_descriptor_artifact_collections(
        first=first_verified,
        second=second_verified,
    )
    assert comparison.scientifically_equivalent is True
    assert first_verified.teacher_forward_count == 0
    assert second_verified.teacher_forward_count == 0


def test_compare_descriptor_artifact_collections_ignores_file_byte_differences_but_catches_scientific_drift(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    manifest_path = descriptor_fixture["cache_root"] / "teacher_cache_manifest.json"
    manifest_path.write_text(
        json.dumps(fixtures.production_like_manifest(descriptor_fixture), sort_keys=True),
        encoding="utf-8",
    )
    subject.materialize_descriptor_artifact_collection(
        config=descriptor_config,
        teacher_cache_manifest_path=manifest_path,
        teacher_cache_root=descriptor_fixture["cache_root"],
        output_run_dir=tmp_path / "run-one",
    )
    subject.materialize_descriptor_artifact_collection(
        config=descriptor_config,
        teacher_cache_manifest_path=manifest_path,
        teacher_cache_root=descriptor_fixture["cache_root"],
        output_run_dir=tmp_path / "run-two",
    )
    first_verified = subject.verify_descriptor_artifact_collection(
        config=descriptor_config,
        run_dir=tmp_path / "run-one",
    )
    second_verified = subject.verify_descriptor_artifact_collection(
        config=descriptor_config,
        run_dir=tmp_path / "run-two",
    )
    comparison = subject.compare_descriptor_artifact_collections(
        first=first_verified,
        second=second_verified,
    )
    assert comparison.scientifically_equivalent is True
    descriptor_file = next((tmp_path / "run-two" / "descriptors").glob("*.pt"))
    with descriptor_file.open("ab") as handle:
        handle.write(b"\x00nonscientific-byte-drift")
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_RECORD_FILE_HASH_MISMATCH"):
        subject.verify_descriptor_artifact_collection(
            config=descriptor_config,
            run_dir=tmp_path / "run-two",
        )


# --------------------------------------------------------------------------------
# Contract gap 1: non-self-referential file-hash placement
# --------------------------------------------------------------------------------


def test_write_descriptor_record_atomic_separates_file_hash_from_payload(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    record = subject.reconstruct_descriptor_record(
        **_reconstruct_kwargs(descriptor_fixture, descriptor_config)
    )
    destination = tmp_path / "descriptors" / f"{record['stable_sample_id']}.pt"
    destination.parent.mkdir(parents=True)
    entry = subject.write_descriptor_record_atomic(destination, record)
    assert entry.descriptor_record_file_sha256
    assert entry.descriptor_record_scientific_sha256 == record[
        "descriptor_record_scientific_sha256"
    ]
    loaded = torch.load(destination, map_location="cpu", weights_only=True)
    assert set(loaded) == {"scientific_record", "descriptor_record_scientific_sha256"}
    assert "descriptor_record_file_sha256" not in loaded
    assert "descriptor_record_file_sha256" not in loaded["scientific_record"]
    assert entry.descriptor_record_file_sha256 == hashlib.sha256(
        destination.read_bytes()
    ).hexdigest()


def test_verify_persisted_descriptor_rejects_file_hash_mismatch(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    record = subject.reconstruct_descriptor_record(
        **_reconstruct_kwargs(descriptor_fixture, descriptor_config)
    )
    destination = tmp_path / "descriptors" / f"{record['stable_sample_id']}.pt"
    destination.parent.mkdir(parents=True)
    entry = subject.write_descriptor_record_atomic(destination, record)
    corrupted = subject.PersistedDescriptorEntry(
        stable_sample_id=entry.stable_sample_id,
        relative_record_path=entry.relative_record_path,
        descriptor_record_scientific_sha256=entry.descriptor_record_scientific_sha256,
        descriptor_record_file_sha256="0" * 64,
        verification_status=entry.verification_status,
    )
    with pytest.raises(
        subject.DescriptorArtifactsError, match="B2_DESC_RECORD_FILE_HASH_MISMATCH"
    ):
        subject.verify_persisted_descriptor_entry(
            run_dir=tmp_path, entry=corrupted, config=descriptor_config
        )


def test_verify_persisted_descriptor_rejects_correct_scientific_wrong_file_bytes(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    record = subject.reconstruct_descriptor_record(
        **_reconstruct_kwargs(descriptor_fixture, descriptor_config)
    )
    destination = tmp_path / "descriptors" / f"{record['stable_sample_id']}.pt"
    destination.parent.mkdir(parents=True)
    entry = subject.write_descriptor_record_atomic(destination, record)
    with destination.open("ab") as handle:
        handle.write(b"\x00tail-corruption")
    with pytest.raises(
        subject.DescriptorArtifactsError, match="B2_DESC_RECORD_FILE_HASH_MISMATCH"
    ):
        subject.verify_persisted_descriptor_entry(
            run_dir=tmp_path, entry=entry, config=descriptor_config
        )


def test_descriptor_scientific_hash_rejects_file_hash_inside_payload(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    record = subject.reconstruct_descriptor_record(
        **_reconstruct_kwargs(descriptor_fixture, descriptor_config)
    )
    baseline = subject.descriptor_record_scientific_sha256(record)
    poisoned = dict(record)
    poisoned["descriptor_record_file_sha256"] = "a" * 64
    assert subject.descriptor_record_scientific_sha256(poisoned) == baseline
    content = subject.descriptor_record_scientific_content(poisoned)
    assert "descriptor_record_file_sha256" not in content


def test_write_normalization_statistics_atomic_separates_file_hash(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    training_records = _training_records(descriptor_fixture, descriptor_config)
    statistics = subject.compute_training_normalization_statistics(
        training_records, config=descriptor_config
    )
    destination = tmp_path / "normalization_statistics.pt"
    entry = subject.write_normalization_statistics_atomic(destination, statistics)
    loaded = torch.load(destination, map_location="cpu", weights_only=True)
    assert set(loaded) == {
        "scientific_statistics_record",
        "normalization_statistics_scientific_sha256",
    }
    assert "normalization_statistics_file_sha256" not in loaded
    assert entry.normalization_statistics_file_sha256 == hashlib.sha256(
        destination.read_bytes()
    ).hexdigest()


def test_verify_normalization_statistics_rejects_file_hash_mismatch(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    training_records = _training_records(descriptor_fixture, descriptor_config)
    statistics = subject.compute_training_normalization_statistics(
        training_records, config=descriptor_config
    )
    destination = tmp_path / "normalization_statistics.pt"
    entry = subject.write_normalization_statistics_atomic(destination, statistics)
    bad = subject.PersistedNormalizationEntry(
        relative_path=entry.relative_path,
        normalization_statistics_scientific_sha256=(
            entry.normalization_statistics_scientific_sha256
        ),
        normalization_statistics_file_sha256="1" * 64,
    )
    with pytest.raises(
        subject.DescriptorArtifactsError,
        match="B2_DESC_NORMALIZATION_FILE_HASH_MISMATCH",
    ):
        subject.verify_persisted_normalization_entry(run_dir=tmp_path, entry=bad)


def test_final_manifest_receipt_must_agree(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    training_records = _training_records(descriptor_fixture, descriptor_config)
    statistics = subject.compute_training_normalization_statistics(
        training_records, config=descriptor_config
    )
    all_records = _all_reconstructed_records(descriptor_fixture, descriptor_config)
    entries = [
        subject.write_descriptor_record_atomic(
            tmp_path / subject.descriptor_relative_path(record["stable_sample_id"]),
            record,
        )
        for record in all_records
    ]
    normalization = subject.write_normalization_statistics_atomic(
        tmp_path / "normalization_statistics.pt",
        statistics,
    )
    manifest = subject.build_descriptor_artifacts_manifest(
        config=descriptor_config,
        records=all_records,
        statistics=statistics,
        sample_entries=entries,
        normalization_entry=normalization,
    )
    subject.write_final_manifest_with_receipt_atomic(tmp_path, manifest)
    assert (tmp_path / "final_manifest.json").is_file()
    assert (tmp_path / "final_manifest.json.sha256").is_file()
    subject.verify_final_manifest_receipt(tmp_path)

    receipt = tmp_path / "final_manifest.json.sha256"
    receipt.write_text("0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(
        subject.DescriptorArtifactsError, match="B2_DESC_MANIFEST_RECEIPT_MISMATCH"
    ):
        subject.verify_final_manifest_receipt(tmp_path)


# --------------------------------------------------------------------------------
# Contract gap 2: collection-level scientific identities
# --------------------------------------------------------------------------------


def test_collection_scientific_identity_is_permutation_invariant(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    all_records = _all_reconstructed_records(descriptor_fixture, descriptor_config)
    statistics = subject.compute_training_normalization_statistics(
        [r for r in all_records if r["split_membership"] == "training"],
        config=descriptor_config,
    )
    forward = subject.descriptor_collection_scientific_sha256(
        records=all_records, statistics=statistics, config=descriptor_config
    )
    reversed_hash = subject.descriptor_collection_scientific_sha256(
        records=list(reversed(all_records)),
        statistics=statistics,
        config=descriptor_config,
    )
    assert forward == reversed_hash


def test_collection_identity_changes_when_one_descriptor_hash_changes(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    all_records = _all_reconstructed_records(descriptor_fixture, descriptor_config)
    statistics = subject.compute_training_normalization_statistics(
        [r for r in all_records if r["split_membership"] == "training"],
        config=descriptor_config,
    )
    baseline = subject.descriptor_collection_scientific_sha256(
        records=all_records, statistics=statistics, config=descriptor_config
    )
    mutated = copy.deepcopy(all_records)
    mutated[0]["descriptor_by_depth"][12] = mutated[0]["descriptor_by_depth"][12] + 1.0
    mutated[0]["descriptor_record_scientific_sha256"] = (
        subject.descriptor_record_scientific_sha256(mutated[0])
    )
    changed = subject.descriptor_collection_scientific_sha256(
        records=mutated, statistics=statistics, config=descriptor_config
    )
    assert changed != baseline


def test_sample_coverage_identity_changes_when_membership_changes(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    all_records = _all_reconstructed_records(descriptor_fixture, descriptor_config)
    baseline = subject.descriptor_sample_coverage_sha256(all_records)
    mutated = copy.deepcopy(all_records)
    training = next(r for r in mutated if r["split_membership"] == "training")
    calibration = next(r for r in mutated if r["split_membership"] == "calibration")
    training["split_membership"], calibration["split_membership"] = (
        "calibration",
        "training",
    )
    assert subject.descriptor_sample_coverage_sha256(mutated) != baseline


def test_normalization_training_coverage_changes_when_training_record_changes(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    training_records = _training_records(descriptor_fixture, descriptor_config)
    baseline_stats = subject.compute_training_normalization_statistics(
        training_records, config=descriptor_config
    )
    baseline = baseline_stats["normalization_training_coverage_sha256"]
    mutated = copy.deepcopy(training_records)
    mutated[0]["descriptor_by_depth"][12] = mutated[0]["descriptor_by_depth"][12] + 2.0
    mutated[0]["descriptor_record_scientific_sha256"] = (
        subject.descriptor_record_scientific_sha256(mutated[0])
    )
    changed_stats = subject.compute_training_normalization_statistics(
        mutated, config=descriptor_config
    )
    assert changed_stats["normalization_training_coverage_sha256"] != baseline
    assert (
        changed_stats["normalization_statistics_scientific_sha256"]
        != baseline_stats["normalization_statistics_scientific_sha256"]
    )


def test_collection_and_coverage_identities_ignore_paths_and_timestamps(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    all_records = _all_reconstructed_records(descriptor_fixture, descriptor_config)
    statistics = subject.compute_training_normalization_statistics(
        [r for r in all_records if r["split_membership"] == "training"],
        config=descriptor_config,
    )
    baseline_collection = subject.descriptor_collection_scientific_sha256(
        records=all_records, statistics=statistics, config=descriptor_config
    )
    baseline_coverage = subject.descriptor_sample_coverage_sha256(all_records)
    poisoned_records = []
    for record in all_records:
        row = dict(record)
        row["absolute_output_path"] = "/tmp/elsewhere"
        row["timestamp"] = "2099-01-01T00:00:00Z"
        row["git_branch"] = "noise"
        poisoned_records.append(row)
    poisoned_stats = dict(statistics)
    poisoned_stats["output_path"] = "/tmp/stats"
    poisoned_stats["timestamp"] = "2099-01-01T00:00:00Z"
    assert (
        subject.descriptor_collection_scientific_sha256(
            records=poisoned_records,
            statistics=poisoned_stats,
            config=descriptor_config,
        )
        == baseline_collection
    )
    assert subject.descriptor_sample_coverage_sha256(poisoned_records) == baseline_coverage


# --------------------------------------------------------------------------------
# Contract gap 3: frozen normalization mathematics
# --------------------------------------------------------------------------------


def test_normalization_uses_population_std_not_sample_std(
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    # Three training IDs; one feature takes values 1, 2, 3 → pop std = sqrt(2/3).
    values = (1.0, 2.0, 3.0)
    records = []
    for index, value in enumerate(values):
        stable_id = f"{index:064x}"
        descriptor_by_depth = {
            12: torch.zeros(1, 2, DESCRIPTOR_DIM, dtype=torch.float32),
            18: torch.zeros(1, 3, DESCRIPTOR_DIM, dtype=torch.float32),
            24: torch.zeros(1, 4, DESCRIPTOR_DIM, dtype=torch.float32),
        }
        descriptor_by_depth[12][0, 0, 0] = value
        record = _hand_built_descriptor_record(
            {"descriptor_contract": {
                "descriptor_contract_version": 1,
                "extractor_configuration_sha256": "a" * 64,
                "descriptor_implementation_sha256": "b" * 64,
            }, "teacher_cache_scientific_sha256": "c" * 64, "checkpoint_sha256": "d" * 64},
            descriptor_config,
            stable_sample_id=stable_id,
            descriptor_by_depth=descriptor_by_depth,
            descriptor_extractor_config_sha256="a" * 64,
            descriptor_extractor_implementation_sha256="b" * 64,
            teacher_cache_scientific_sha256="c" * 64,
            checkpoint_sha256="d" * 64,
        )
        # Bypass full 16-count by using a dedicated helper path that accepts synthetic
        # mini-batches only through the math unit under test.
        records.append(record)

    # Pad to exactly 16 with constant zeros so the Gate-C count contract still holds.
    while len(records) < 16:
        stable_id = f"{len(records):064x}"
        records.append(
            _hand_built_descriptor_record(
                {"descriptor_contract": {
                    "descriptor_contract_version": 1,
                    "extractor_configuration_sha256": "a" * 64,
                    "descriptor_implementation_sha256": "b" * 64,
                }, "teacher_cache_scientific_sha256": "c" * 64, "checkpoint_sha256": "d" * 64},
                descriptor_config,
                stable_sample_id=stable_id,
                descriptor_extractor_config_sha256="a" * 64,
                descriptor_extractor_implementation_sha256="b" * 64,
                teacher_cache_scientific_sha256="c" * 64,
                checkpoint_sha256="d" * 64,
            )
        )
    statistics = subject.compute_training_normalization_statistics(
        records, config=descriptor_config
    )
    feature = statistics["axes"][12]["layers"][0]["features"][0]
    assert feature["descriptor_feature_name"] == FEATURE_NAMES[0]
    # Across 16 samples: three non-zero starters + thirteen zeros at this coordinate.
    # Mean and pop-std are computed over all 16 training rows.
    expected_values = list(values) + [0.0] * 13
    mean = sum(expected_values) / 16
    pop_var = sum((v - mean) ** 2 for v in expected_values) / 16
    pop_std = pop_var**0.5
    sample_std = (sum((v - mean) ** 2 for v in expected_values) / 15) ** 0.5
    assert feature["std"] == pytest.approx(pop_std)
    assert feature["std"] != pytest.approx(sample_std)
    assert feature["mean"] == pytest.approx(mean)
    assert isinstance(feature["mean"], float)
    assert isinstance(feature["std"], float)


def test_normalization_axes_bind_candidate_layer_id_not_only_position(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    training_records = _training_records(descriptor_fixture, descriptor_config)
    statistics = subject.compute_training_normalization_statistics(
        training_records, config=descriptor_config
    )
    for depth in PREDICTION_DEPTHS:
        layers = statistics["axes"][depth]["layers"]
        expected_ids = [layer for layer in CANDIDATE_LAYERS if layer <= depth]
        assert [row["candidate_layer_id"] for row in layers] == expected_ids
        for position, row in enumerate(layers):
            assert row["candidate_layer_position"] == position
            assert row["candidate_layer_id"] == expected_ids[position]
            assert [f["descriptor_feature_name"] for f in row["features"]] == list(
                FEATURE_NAMES
            )


def test_normalization_excludes_invalid_layers_from_statistics(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    training_records = _training_records(descriptor_fixture, descriptor_config)
    mutated = copy.deepcopy(training_records)
    for record in mutated:
        # Invalidate the first layer at depth 12 for every training sample.
        record["valid_layer_mask_by_depth"][12] = [False, True]
        record["descriptor_record_scientific_sha256"] = (
            subject.descriptor_record_scientific_sha256(record)
        )
    statistics = subject.compute_training_normalization_statistics(
        mutated, config=descriptor_config
    )
    layer_ids = [row["candidate_layer_id"] for row in statistics["axes"][12]["layers"]]
    assert 6 not in layer_ids
    assert 12 in layer_ids


def test_apply_frozen_normalization_is_float64_then_float32_and_immutable(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    training_records = _training_records(descriptor_fixture, descriptor_config)
    statistics = subject.compute_training_normalization_statistics(
        training_records, config=descriptor_config
    )
    probe = copy.deepcopy(training_records[0])
    original_ids = {
        depth: id(probe["descriptor_by_depth"][depth]) for depth in PREDICTION_DEPTHS
    }
    original_values = {
        depth: probe["descriptor_by_depth"][depth].detach().clone()
        for depth in PREDICTION_DEPTHS
    }
    normalized = subject.apply_frozen_normalization(probe, statistics)
    for depth in PREDICTION_DEPTHS:
        assert id(probe["descriptor_by_depth"][depth]) == original_ids[depth]
        assert torch.equal(probe["descriptor_by_depth"][depth], original_values[depth])
        tensor = normalized["descriptor_by_depth"][depth]
        assert tensor.dtype == torch.float32
        assert bool(torch.isfinite(tensor).all())
    # Determinism: identical inputs → identical float32 outputs.
    again = subject.apply_frozen_normalization(probe, statistics)
    for depth in PREDICTION_DEPTHS:
        assert torch.equal(
            normalized["descriptor_by_depth"][depth],
            again["descriptor_by_depth"][depth],
        )


def test_normalization_float64_accumulation_is_documented_in_statistics_dtype(
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    training_records = _training_records(descriptor_fixture, descriptor_config)
    statistics = subject.compute_training_normalization_statistics(
        training_records, config=descriptor_config
    )
    assert statistics["statistics_dtype"] == "float64"
    assert statistics["application_output_dtype"] == "float32"
    assert statistics["standard_deviation_ddof"] == 0


# --------------------------------------------------------------------------------
# B2-03A Story 11: authoritative run-root binding and exact run-relative file sets
# --------------------------------------------------------------------------------


def _materialized_collection(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> tuple[subject.DescriptorCollectionResult, Path]:
    manifest_path = descriptor_fixture["cache_root"] / "teacher_cache_manifest.json"
    manifest_path.write_text(
        json.dumps(fixtures.production_like_manifest(descriptor_fixture), sort_keys=True),
        encoding="utf-8",
    )
    run_dir = tmp_path / "materialized-run"
    result = subject.materialize_descriptor_artifact_collection(
        config=descriptor_config,
        teacher_cache_manifest_path=manifest_path,
        teacher_cache_root=descriptor_fixture["cache_root"],
        output_run_dir=run_dir,
    )
    return result, run_dir


def test_verify_descriptor_artifact_collection_rejects_descriptor_parent_escape(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    _, run_dir = _materialized_collection(tmp_path, descriptor_fixture, descriptor_config)
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    manifest_path = run_dir / "final_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["samples"][0]["relative_record_path"] = "../outside.pt"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    (run_dir / "final_manifest.json.sha256").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_"):
        subject.verify_descriptor_artifact_collection(config=descriptor_config, run_dir=run_dir)


def test_verify_descriptor_artifact_collection_rejects_absolute_descriptor_path(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    _, run_dir = _materialized_collection(tmp_path, descriptor_fixture, descriptor_config)
    outside = (tmp_path / "absolute.pt").resolve()
    outside.write_bytes(b"outside")
    manifest_path = run_dir / "final_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["samples"][0]["relative_record_path"] = str(outside)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    (run_dir / "final_manifest.json.sha256").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_"):
        subject.verify_descriptor_artifact_collection(config=descriptor_config, run_dir=run_dir)


def test_verify_descriptor_artifact_collection_rejects_symlink_escape(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    _, run_dir = _materialized_collection(tmp_path, descriptor_fixture, descriptor_config)
    samples = json.loads((run_dir / "final_manifest.json").read_text(encoding="utf-8"))["samples"]
    stable_id = samples[0]["stable_sample_id"]
    target = tmp_path / "escaped-target.pt"
    target.write_text("escaped\n", encoding="utf-8")
    path = run_dir / "descriptors" / f"{stable_id}.pt"
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_"):
        subject.verify_descriptor_artifact_collection(config=descriptor_config, run_dir=run_dir)


def test_verify_descriptor_artifact_collection_rejects_wrong_nested_basename_match(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    _, run_dir = _materialized_collection(tmp_path, descriptor_fixture, descriptor_config)
    manifest_path = run_dir / "final_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stable_id = manifest["samples"][0]["stable_sample_id"]
    original = run_dir / "descriptors" / f"{stable_id}.pt"
    nested = run_dir / "descriptors" / "wrong" / f"{stable_id}.pt"
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.write_bytes(original.read_bytes())
    original.unlink()
    manifest["samples"][0]["relative_record_path"] = f"descriptors/wrong/{stable_id}.pt"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    (run_dir / "final_manifest.json.sha256").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_"):
        subject.verify_descriptor_artifact_collection(config=descriptor_config, run_dir=run_dir)


def test_verify_descriptor_artifact_collection_rejects_two_records_pointing_to_one_file(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    _, run_dir = _materialized_collection(tmp_path, descriptor_fixture, descriptor_config)
    manifest_path = run_dir / "final_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["samples"][1]["relative_record_path"] = manifest["samples"][0]["relative_record_path"]
    manifest["samples"][1]["descriptor_record_file_sha256"] = manifest["samples"][0][
        "descriptor_record_file_sha256"
    ]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    (run_dir / "final_manifest.json.sha256").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_"):
        subject.verify_descriptor_artifact_collection(config=descriptor_config, run_dir=run_dir)


def test_verify_descriptor_artifact_collection_rejects_normalization_path_outside_root(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    _, run_dir = _materialized_collection(tmp_path, descriptor_fixture, descriptor_config)
    outside = tmp_path / "outside-normalization.pt"
    outside.write_text("outside\n", encoding="utf-8")
    manifest_path = run_dir / "final_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["normalization_statistics_relative_path"] = "../outside-normalization.pt"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    (run_dir / "final_manifest.json.sha256").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_"):
        subject.verify_descriptor_artifact_collection(config=descriptor_config, run_dir=run_dir)


def test_verify_descriptor_artifact_collection_rejects_manifest_receipt_outside_root(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    _, run_dir = _materialized_collection(tmp_path, descriptor_fixture, descriptor_config)
    outside_receipt = tmp_path / "outside.sha256"
    outside_receipt.write_text("0" * 64 + "\n", encoding="utf-8")
    original_receipt = run_dir / "final_manifest.json.sha256"
    original_receipt.unlink()
    original_receipt.symlink_to(outside_receipt)
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_"):
        subject.verify_descriptor_artifact_collection(config=descriptor_config, run_dir=run_dir)


def test_verify_descriptor_artifact_collection_rejects_unexpected_nested_pt(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    _, run_dir = _materialized_collection(tmp_path, descriptor_fixture, descriptor_config)
    nested = run_dir / "descriptors" / "nested" / ("f" * 64 + ".pt")
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_"):
        subject.verify_descriptor_artifact_collection(config=descriptor_config, run_dir=run_dir)


def test_verify_descriptor_artifact_collection_accepts_exact_valid_run_relative_file_set(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    result, run_dir = _materialized_collection(tmp_path, descriptor_fixture, descriptor_config)
    verified = subject.verify_descriptor_artifact_collection(
        config=descriptor_config,
        run_dir=run_dir,
    )
    assert verified.teacher_forward_count == 0
    assert verified.manifest["descriptor_collection_scientific_sha256"] == result.manifest[
        "descriptor_collection_scientific_sha256"
    ]


def test_resolve_run_relative_artifact_normalizes_formatting_deterministically(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    _, run_dir = _materialized_collection(tmp_path, descriptor_fixture, descriptor_config)
    stable_id = next(iter(json.loads((run_dir / "final_manifest.json").read_text(encoding="utf-8"))["planned_stable_sample_ids"]))
    first = subject.resolve_run_relative_artifact(
        run_dir=run_dir,
        relative_path=f"descriptors/./{stable_id}.pt",
        expected_kind="descriptor record",
    )
    second = subject.resolve_run_relative_artifact(
        run_dir=run_dir,
        relative_path=f"descriptors/{stable_id}.pt",
        expected_kind="descriptor record",
    )
    assert first == second


# --------------------------------------------------------------------------------
# B2-03A Story 12: disk-authoritative teacher-cache provenance parity
# --------------------------------------------------------------------------------


def test_common_disk_loader_rejects_manifest_path_outside_cache_root(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    outside_manifest = tmp_path / "teacher_cache_manifest.json"
    outside_manifest.write_text(
        json.dumps(fixtures.production_like_manifest(descriptor_fixture), sort_keys=True),
        encoding="utf-8",
    )
    isolated_root = tmp_path / "isolated-root"
    isolated_root.mkdir()
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_CACHE_MANIFEST_OUTSIDE_ROOT"):
        subject.load_and_validate_accepted_teacher_cache_from_disk(
            config=descriptor_config,
            teacher_cache_manifest_path=outside_manifest,
            teacher_cache_root=isolated_root,
        )


def test_common_disk_loader_rejects_record_drift_even_when_manifest_checkpoint_matches(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    manifest = fixtures.production_like_manifest(descriptor_fixture)
    stable_id = manifest["samples"][0]["stable_sample_id"]

    def mutate_tensor(record: dict[str, Any]) -> dict[str, Any]:
        name = next(key for key in sorted(record["tensors"]) if key.startswith("causal_map:"))
        tensor = record["tensors"][name]["tensor"].clone()
        tensor[0, 0, 0, 0] += 1.0
        record["tensors"][name] = dict(record["tensors"][name])
        record["tensors"][name]["tensor"] = tensor
        record["tensors"][name]["digest"] = cache_mod.canonical_tensor_digest(
            name, tensor, tuple(record["tensors"][name]["dimension_semantics"])
        )
        return record

    fixtures.rewrite_sample_record(descriptor_fixture, manifest, stable_id, mutate_tensor)
    manifest["checkpoint_sha256"] = descriptor_config.expected_checkpoint_sha256
    manifest_path = descriptor_fixture["cache_root"] / "teacher_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_CACHE_SCIENTIFIC_HASH_MISMATCH"):
        subject.load_and_validate_accepted_teacher_cache_from_disk(
            config=descriptor_config,
            teacher_cache_manifest_path=manifest_path,
            teacher_cache_root=descriptor_fixture["cache_root"],
        )


def test_common_disk_loader_rejects_profile_mismatch_in_manifest(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    manifest = fixtures.production_like_manifest(descriptor_fixture)
    manifest["execution_profile_sha256"] = "7" * 64
    manifest_path = descriptor_fixture["cache_root"] / "teacher_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_COLLECTION_PROVENANCE_MISMATCH"):
        subject.load_and_validate_accepted_teacher_cache_from_disk(
            config=descriptor_config,
            teacher_cache_manifest_path=manifest_path,
            teacher_cache_root=descriptor_fixture["cache_root"],
        )


def test_common_disk_loader_rejects_record_candidate_layer_drift(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    manifest = fixtures.production_like_manifest(descriptor_fixture)
    stable_id = manifest["samples"][0]["stable_sample_id"]

    def mutate_candidate_layers(record: dict[str, Any]) -> dict[str, Any]:
        record["candidate_layers"] = [6, 12, 18]
        return record

    fixtures.rewrite_sample_record(
        descriptor_fixture, manifest, stable_id, mutate_candidate_layers
    )
    manifest_path = descriptor_fixture["cache_root"] / "teacher_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_COLLECTION_PROVENANCE_MISMATCH"):
        subject.load_and_validate_accepted_teacher_cache_from_disk(
            config=descriptor_config,
            teacher_cache_manifest_path=manifest_path,
            teacher_cache_root=descriptor_fixture["cache_root"],
        )


def test_common_disk_loader_rejects_record_implementation_digest_drift(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    manifest = fixtures.production_like_manifest(descriptor_fixture)
    stable_id = manifest["samples"][0]["stable_sample_id"]

    def mutate_impl(record: dict[str, Any]) -> dict[str, Any]:
        record["descriptor_implementation_sha256"] = "8" * 64
        return record

    fixtures.rewrite_sample_record(descriptor_fixture, manifest, stable_id, mutate_impl)
    manifest_path = descriptor_fixture["cache_root"] / "teacher_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(subject.DescriptorArtifactsError, match="B2_DESC_CONTRACT_IDENTITY_MISMATCH"):
        subject.load_and_validate_accepted_teacher_cache_from_disk(
            config=descriptor_config,
            teacher_cache_manifest_path=manifest_path,
            teacher_cache_root=descriptor_fixture["cache_root"],
        )


def test_common_disk_loader_tracks_source_manifest_file_sha_changes(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    manifest = fixtures.production_like_manifest(descriptor_fixture)
    manifest_path = descriptor_fixture["cache_root"] / "teacher_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    first = subject.load_and_validate_accepted_teacher_cache_from_disk(
        config=descriptor_config,
        teacher_cache_manifest_path=manifest_path,
        teacher_cache_root=descriptor_fixture["cache_root"],
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    second = subject.load_and_validate_accepted_teacher_cache_from_disk(
        config=descriptor_config,
        teacher_cache_manifest_path=manifest_path,
        teacher_cache_root=descriptor_fixture["cache_root"],
    )
    assert first.source_teacher_cache_manifest_file_sha256 != second.source_teacher_cache_manifest_file_sha256
    assert first.accepted.entries == second.accepted.entries


def test_materialize_and_common_disk_loader_share_same_validated_identity(
    tmp_path: Path,
    descriptor_fixture: dict[str, Any],
    descriptor_config: subject.DescriptorArtifactsConfig,
) -> None:
    manifest = fixtures.production_like_manifest(descriptor_fixture)
    manifest_path = descriptor_fixture["cache_root"] / "teacher_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    validated = subject.load_and_validate_accepted_teacher_cache_from_disk(
        config=descriptor_config,
        teacher_cache_manifest_path=manifest_path,
        teacher_cache_root=descriptor_fixture["cache_root"],
    )
    direct = subject.materialize_descriptor_artifact_collection(
        config=descriptor_config,
        teacher_cache_manifest_path=manifest_path,
        teacher_cache_root=descriptor_fixture["cache_root"],
        output_run_dir=tmp_path / "materialized-from-direct-path",
    )
    preload = subject.materialize_descriptor_artifact_collection(
        config=descriptor_config,
        teacher_cache_manifest_path=manifest_path,
        teacher_cache_root=descriptor_fixture["cache_root"],
        output_run_dir=tmp_path / "materialized-from-prevalidated-path",
        validated_teacher_cache=validated,
    )
    assert validated.accepted.manifest["cache_scientific_sha256"] == direct.manifest[
        "expected_teacher_cache_scientific_sha256"
    ]
    assert direct.source_teacher_cache_manifest_file_sha256 == preload.source_teacher_cache_manifest_file_sha256
    assert direct.manifest["source_teacher_cache_manifest_file_sha256"] == preload.manifest[
        "source_teacher_cache_manifest_file_sha256"
    ]

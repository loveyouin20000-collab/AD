"""Deterministic in-memory fixture for B2-04A contribution-target tests.

Test-only. Everything here is arithmetically deterministic (no RNG, no disk, no
dataset access, no teacher forward) and is always marked
``artifact_kind == "test_fixture"`` so production official mode can never accept
it. Do not import this module from production code.

The fixture models the accepted upstream boundary that B2-04A binds against:

* one teacher cache (``teacher A``) providing production-shaped causal maps
  ``[batch, channel, height, width]`` per ``(checkpoint_depth, candidate_layer)``,
  a binary GT mask, and a full-depth teacher reference that is bit-exactly the
  production ``sum_preserving_fusion`` of the depth-24 candidate-layer maps;
* one descriptor collection (``descriptor A`` or ``descriptor B``) that is the
  future DLCM feature anchor and is never recomputed here.

The two descriptor variants model the B2-04B dual-run boundary: Target Run A is
teacher A + descriptor A and Target Run B is teacher A + descriptor B, so both
runs share every teacher-side identity and differ only in descriptor identity.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import rad.phase_b.b2_contribution_targets as targets
from rad.models import dlcm

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKED_CONFIG_PATH = (
    REPO_ROOT / "configs" / "phase_b" / "b2_contribution_targets_gate_c.json"
)

FIXTURE_ARTIFACT_KIND = "test_fixture"
FIXTURE_CANDIDATE_LAYERS: tuple[int, ...] = (6, 12, 18, 24)
FIXTURE_PREDICTION_DEPTHS: tuple[int, ...] = (12, 18, 24)
FIXTURE_MAP_SHAPE: tuple[int, ...] = (1, 1, 12, 12)
FIXTURE_SPLIT_COUNTS: Mapping[str, int] = {"training": 16, "calibration": 8, "evaluation": 8}
FIXTURE_CATEGORIES: tuple[str, ...] = ("bottle", "cable", "capsule", "carpet")
FIXTURE_ANOMALY_TYPES: tuple[str, ...] = ("broken_large", "scratch", "bent_wire", "hole")
DESCRIPTOR_VARIANTS: tuple[str, ...] = ("A", "B")

_TEACHER_RUN_ID = "teacher-a"
_LCG_MULTIPLIER = 6364136223846793005
_LCG_INCREMENT = 1442695040888963407
_LCG_MODULUS = 1 << 64


def _digest(*parts: Any) -> str:
    return hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _deterministic_unit_values(seed_material: str, count: int) -> list[float]:
    """Reproducible ``[0, 1)`` values from a pure-Python LCG seeded by SHA-256."""

    state = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest()[:8], "big")
    values: list[float] = []
    for _ in range(count):
        state = (state * _LCG_MULTIPLIER + _LCG_INCREMENT) % _LCG_MODULUS
        values.append((state >> 11) / float(1 << 53))
    return values


def _deterministic_map(
    stable_sample_id: str,
    depth: int,
    layer: int,
    shape: Sequence[int],
) -> torch.Tensor:
    count = 1
    for size in shape:
        count *= int(size)
    values = _deterministic_unit_values(f"causal_map:{stable_sample_id}:{depth}:{layer}", count)
    return torch.tensor(values, dtype=torch.float32).reshape(tuple(int(s) for s in shape))


def _deterministic_mask(stable_sample_id: str, shape: Sequence[int]) -> torch.Tensor:
    """Binary mask with a deterministic contiguous anomaly block and background."""

    height, width = int(shape[-2]), int(shape[-1])
    offsets = _deterministic_unit_values(f"mask:{stable_sample_id}", 4)
    top = int(offsets[0] * (height // 2))
    left = int(offsets[1] * (width // 2))
    block_height = 1 + int(offsets[2] * max(1, height // 3))
    block_width = 1 + int(offsets[3] * max(1, width // 3))
    mask = torch.zeros(tuple(int(size) for size in shape), dtype=torch.float32)
    mask[..., top : top + block_height, left : left + block_width] = 1.0
    positives = int(mask.sum().item())
    if positives == 0 or positives == int(mask.numel()):
        raise AssertionError(f"fixture mask for {stable_sample_id} is degenerate")
    return mask


def _full_depth_reference(
    maps_by_layer: Mapping[int, torch.Tensor],
    candidate_layers: Sequence[int],
) -> torch.Tensor:
    """Production sum-preserving fusion with equal valid weights over all layers."""

    stacked = torch.stack([maps_by_layer[int(layer)] for layer in candidate_layers], dim=1)
    valid_mask = torch.ones(stacked.shape[:2], dtype=torch.bool)
    weights = valid_mask.to(stacked.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True)
    return dlcm.sum_preserving_fusion(stacked, weights, valid_mask)


@dataclass(frozen=True)
class FixtureSample:
    """One deterministic fixture sample with its upstream teacher/descriptor records."""

    index: int
    stable_sample_id: str
    membership: str
    category: str
    label: int
    anomaly_type: str
    mask_identity: str | None
    maps_by_depth: Mapping[int, Mapping[int, torch.Tensor]]
    mask: torch.Tensor
    full_depth_map: torch.Tensor
    teacher_record: Mapping[str, Any]
    teacher_record_scientific_sha256: str
    descriptor_record: Mapping[str, Any]

    @property
    def is_anomalous(self) -> bool:
        return self.label == 1


@dataclass(frozen=True)
class ContributionTargetFixture:
    """A complete 32-sample hermetic contribution-target input fixture."""

    artifact_kind: str
    descriptor_variant: str
    candidate_layers: tuple[int, ...]
    prediction_depths: tuple[int, ...]
    map_shape: tuple[int, ...]
    split_counts: Mapping[str, int]
    samples: tuple[FixtureSample, ...]
    teacher_cache_scientific_sha256: str
    teacher_cache_sample_coverage_sha256: str
    descriptor_collection_scientific_sha256: str
    split_scientific_sha256: str
    checkpoint_sha256: str
    execution_profile_sha256: str

    def by_membership(self, membership: str) -> tuple[FixtureSample, ...]:
        return tuple(sample for sample in self.samples if sample.membership == membership)

    def sample_by_id(self, stable_sample_id: str) -> FixtureSample:
        for sample in self.samples:
            if sample.stable_sample_id == stable_sample_id:
                return sample
        raise AssertionError(f"unknown fixture sample {stable_sample_id}")


def _memberships() -> tuple[str, ...]:
    ordered: list[str] = []
    for membership in ("training", "calibration", "evaluation"):
        ordered.extend([membership] * int(FIXTURE_SPLIT_COUNTS[membership]))
    return tuple(ordered)


def build_contribution_target_fixture(
    *,
    descriptor_variant: str = "A",
    map_shape: Sequence[int] = FIXTURE_MAP_SHAPE,
    candidate_layers: Sequence[int] = FIXTURE_CANDIDATE_LAYERS,
    prediction_depths: Sequence[int] = FIXTURE_PREDICTION_DEPTHS,
) -> ContributionTargetFixture:
    """Build the deterministic 32-sample (16/8/8) hermetic fixture."""

    if descriptor_variant not in DESCRIPTOR_VARIANTS:
        raise AssertionError(f"unknown descriptor variant {descriptor_variant!r}")
    layers = tuple(int(layer) for layer in candidate_layers)
    depths = tuple(int(depth) for depth in prediction_depths)
    shape = tuple(int(size) for size in map_shape)

    teacher_cache_hash = _digest("teacher-cache", _TEACHER_RUN_ID)
    teacher_coverage_hash = _digest("teacher-cache-coverage", _TEACHER_RUN_ID)
    descriptor_collection_hash = _digest("descriptor-collection", descriptor_variant)
    split_hash = _digest("split-manifest", "b2-tiny-split-v2")
    checkpoint_hash = _digest("checkpoint", _TEACHER_RUN_ID)
    profile_hash = _digest("execution-profile", "cpu-deterministic")

    samples: list[FixtureSample] = []
    for index, membership in enumerate(_memberships()):
        stable_sample_id = _digest("b2-04a-fixture-sample", index)
        label = 1 if index % 2 == 0 else 0
        category = FIXTURE_CATEGORIES[index % len(FIXTURE_CATEGORIES)]
        anomaly_type = (
            FIXTURE_ANOMALY_TYPES[index % len(FIXTURE_ANOMALY_TYPES)] if label == 1 else "good"
        )
        maps_by_depth = {
            depth: {
                layer: _deterministic_map(stable_sample_id, depth, layer, shape)
                for layer in layers
                if layer <= depth
            }
            for depth in depths
        }
        mask_identity = (
            f"fixture/{category}/{anomaly_type}/{index:03d}_mask.png" if label == 1 else None
        )
        mask = (
            _deterministic_mask(stable_sample_id, shape)
            if label == 1
            else torch.zeros(shape, dtype=torch.float32)
        )
        full_depth_map = _full_depth_reference(maps_by_depth[max(depths)], layers)
        teacher_record = {
            "stable_sample_id": stable_sample_id,
            "membership": membership,
            "category": category,
            "image_label": label,
            "anomaly_type": anomaly_type,
            "candidate_layers": list(layers),
            "prediction_depths": list(depths),
            "mask_identity": mask_identity,
            "split_scientific_sha256": split_hash,
            "checkpoint_sha256": checkpoint_hash,
            "execution_profile_sha256": profile_hash,
        }
        teacher_record_hash = _digest("teacher-record", _TEACHER_RUN_ID, stable_sample_id)
        descriptor_record = {
            "stable_sample_id": stable_sample_id,
            "split_membership": membership,
            "candidate_layers": list(layers),
            "prediction_depths": list(depths),
            "source_teacher_record_scientific_sha256": teacher_record_hash,
            "teacher_cache_scientific_sha256": teacher_cache_hash,
            "split_scientific_sha256": split_hash,
            "checkpoint_sha256": checkpoint_hash,
            "execution_profile_sha256": profile_hash,
            "descriptor_record_scientific_sha256": _digest(
                "descriptor-record", descriptor_variant, stable_sample_id
            ),
        }
        samples.append(
            FixtureSample(
                index=index,
                stable_sample_id=stable_sample_id,
                membership=membership,
                category=category,
                label=label,
                anomaly_type=anomaly_type,
                mask_identity=mask_identity,
                maps_by_depth=maps_by_depth,
                mask=mask,
                full_depth_map=full_depth_map,
                teacher_record=teacher_record,
                teacher_record_scientific_sha256=teacher_record_hash,
                descriptor_record=descriptor_record,
            )
        )

    return ContributionTargetFixture(
        artifact_kind=FIXTURE_ARTIFACT_KIND,
        descriptor_variant=descriptor_variant,
        candidate_layers=layers,
        prediction_depths=depths,
        map_shape=shape,
        split_counts=dict(FIXTURE_SPLIT_COUNTS),
        samples=tuple(samples),
        teacher_cache_scientific_sha256=teacher_cache_hash,
        teacher_cache_sample_coverage_sha256=teacher_coverage_hash,
        descriptor_collection_scientific_sha256=descriptor_collection_hash,
        split_scientific_sha256=split_hash,
        checkpoint_sha256=checkpoint_hash,
        execution_profile_sha256=profile_hash,
    )


def fixture_calibration_samples(
    fixture: ContributionTargetFixture,
) -> tuple[targets.GtCalibrationSample, ...]:
    """The source-training GT calibration inputs for this fixture."""

    return tuple(
        targets.GtCalibrationSample(
            stable_sample_id=sample.stable_sample_id,
            membership=sample.membership,
            maps_by_depth=sample.maps_by_depth,
        )
        for sample in fixture.by_membership("training")
    )


def fit_fixture_calibration(fixture: ContributionTargetFixture) -> targets.GtMapCalibration:
    return targets.fit_gt_map_calibration(
        fixture_calibration_samples(fixture),
        candidate_layers=fixture.candidate_layers,
        prediction_depths=fixture.prediction_depths,
    )


def build_fixture_calibration_artifact(
    fixture: ContributionTargetFixture,
) -> dict[str, Any]:
    """Build the fixture's GT map calibration artifact with its scientific hash."""

    calibration = fit_fixture_calibration(fixture)
    training = fixture.by_membership("training")
    return targets.build_gt_map_calibration_artifact(
        calibration,
        source_teacher_record_scientific_sha256_by_id={
            sample.stable_sample_id: sample.teacher_record_scientific_sha256
            for sample in training
        },
        teacher_cache_scientific_sha256=fixture.teacher_cache_scientific_sha256,
        teacher_cache_sample_coverage_sha256=fixture.teacher_cache_sample_coverage_sha256,
        descriptor_collection_scientific_sha256=fixture.descriptor_collection_scientific_sha256,
        split_scientific_sha256=fixture.split_scientific_sha256,
        checkpoint_sha256=fixture.checkpoint_sha256,
        execution_profile_sha256=fixture.execution_profile_sha256,
        expected_training_count=int(fixture.split_counts["training"]),
        artifact_kind=fixture.artifact_kind,
    )


def fixture_upstream(
    fixture: ContributionTargetFixture,
    sample: FixtureSample,
) -> targets.UpstreamTargetIdentities:
    """Bind one fixture sample's teacher and descriptor identities."""

    return targets.bind_upstream_identities(
        teacher_record=sample.teacher_record,
        teacher_record_scientific_sha256=sample.teacher_record_scientific_sha256,
        teacher_cache_scientific_sha256=fixture.teacher_cache_scientific_sha256,
        teacher_cache_sample_coverage_sha256=fixture.teacher_cache_sample_coverage_sha256,
        descriptor_record=sample.descriptor_record,
        descriptor_collection_scientific_sha256=fixture.descriptor_collection_scientific_sha256,
        candidate_layers=fixture.candidate_layers,
        prediction_depths=fixture.prediction_depths,
    )


def fixture_mask_provenance(sample: FixtureSample) -> targets.MaskProvenance:
    return targets.MaskProvenance(
        mask_identity=sample.mask_identity,
        mask_source=(
            "production_gt_mask" if sample.is_anomalous else "normal_all_zero_mask"
        ),
    )


def fixture_teacher_reference_provenance(
    fixture: ContributionTargetFixture,
    sample: FixtureSample,
) -> targets.TeacherReferenceProvenance:
    reconstructed = targets.reconstruct_full_depth_teacher(
        sample.maps_by_depth[max(fixture.prediction_depths)],
        candidate_layers=fixture.candidate_layers,
    )
    targets.verify_full_depth_teacher_bitexact(reconstructed, sample.full_depth_map)
    return targets.TeacherReferenceProvenance(
        cached_full_depth_map_digest=targets.full_depth_map_digest(sample.full_depth_map),
        reconstruction_verified=True,
        source_candidate_layers=fixture.candidate_layers,
    )


def build_fixture_record(
    fixture: ContributionTargetFixture,
    sample: FixtureSample,
    calibration_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one contribution-target record for a fixture sample."""

    return targets.build_contribution_target_record(
        sample=targets.ContributionTargetSample(
            stable_sample_id=sample.stable_sample_id,
            split_membership=sample.membership,
            category=sample.category,
            label=sample.label,
            anomaly_type=sample.anomaly_type,
            maps_by_depth=sample.maps_by_depth,
            mask=sample.mask,
            teacher_reference_map=sample.full_depth_map,
        ),
        calibration_artifact=calibration_artifact,
        upstream=fixture_upstream(fixture, sample),
        mask_provenance=fixture_mask_provenance(sample),
        teacher_reference_provenance=fixture_teacher_reference_provenance(fixture, sample),
        candidate_layers=fixture.candidate_layers,
        prediction_depths=fixture.prediction_depths,
        artifact_kind=fixture.artifact_kind,
    )


_RECORD_CACHE: dict[tuple[str, tuple[int, ...]], tuple[dict[str, Any], ...]] = {}
_CALIBRATION_CACHE: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}


def _cache_key(fixture: ContributionTargetFixture) -> tuple[str, tuple[int, ...]]:
    return (fixture.descriptor_variant, fixture.map_shape)


def fixture_calibration_artifact(fixture: ContributionTargetFixture) -> dict[str, Any]:
    """Cached GT map calibration artifact (deep-copied per call)."""

    key = _cache_key(fixture)
    if key not in _CALIBRATION_CACHE:
        _CALIBRATION_CACHE[key] = build_fixture_calibration_artifact(fixture)
    return copy.deepcopy(_CALIBRATION_CACHE[key])


def build_fixture_records(
    fixture: ContributionTargetFixture,
) -> tuple[dict[str, Any], ...]:
    """All 32 contribution-target records, cached and deep-copied per call."""

    key = _cache_key(fixture)
    if key not in _RECORD_CACHE:
        artifact = fixture_calibration_artifact(fixture)
        _RECORD_CACHE[key] = tuple(
            build_fixture_record(fixture, sample, artifact) for sample in fixture.samples
        )
    return tuple(copy.deepcopy(record) for record in _RECORD_CACHE[key])


def records_by_membership(
    records: Sequence[Mapping[str, Any]],
    membership: str,
) -> tuple[Mapping[str, Any], ...]:
    return tuple(row for row in records if row["split_membership"] == membership)


def build_fixture_normalization(
    fixture: ContributionTargetFixture,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return targets.compute_shapley_normalization(
        records_by_membership(records, "training"),
        candidate_layers=fixture.candidate_layers,
        prediction_depths=fixture.prediction_depths,
        expected_training_count=int(fixture.split_counts["training"]),
        artifact_kind=fixture.artifact_kind,
    )


def tracked_config_payload() -> dict[str, Any]:
    """Deep-copied tracked Gate-C configuration object."""

    return json.loads(TRACKED_CONFIG_PATH.read_text(encoding="utf-8"))


def write_config(
    tmp_path: Path,
    payload: Mapping[str, Any],
    *,
    name: str = "config.json",
) -> Path:
    """Write one JSON configuration under ``tmp_path`` and return its path."""

    path = Path(tmp_path) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def write_controlled_official_config(tmp_path: Path, **overrides: Any) -> Path:
    """Write a non-tracked config that may enable official materialization in tests."""

    return write_config(
        tmp_path,
        controlled_official_config_payload(**overrides),
        name="controlled_official.json",
    )


def controlled_official_config_payload(**overrides: Any) -> dict[str, Any]:
    """Non-tracked Gate-C payload that may enable official materialization."""

    payload = tracked_config_payload()
    payload["configuration_id"] = "b2_contribution_targets_controlled_official"
    payload["official_materialization_enabled"] = True
    payload["expected_input_artifact_kind"] = FIXTURE_ARTIFACT_KIND
    payload.update(overrides)
    return payload


def fixture_input_bundle(
    fixture: ContributionTargetFixture,
    *,
    samples: Sequence[FixtureSample] | None = None,
) -> targets.ContributionInputBundle:
    """Convert an in-memory fixture into the shared ``ContributionInputBundle``."""

    selected = tuple(fixture.samples if samples is None else samples)
    return targets.ContributionInputBundle(
        artifact_kind=fixture.artifact_kind,
        candidate_layers=fixture.candidate_layers,
        prediction_depths=fixture.prediction_depths,
        samples=tuple(
            targets.ContributionInputSample(
                stable_sample_id=sample.stable_sample_id,
                split_membership=sample.membership,
                category=sample.category,
                label=sample.label,
                anomaly_type=sample.anomaly_type,
                mask_identity=sample.mask_identity,
                maps_by_depth=sample.maps_by_depth,
                mask=sample.mask,
                full_depth_map=sample.full_depth_map,
                teacher_record=sample.teacher_record,
                teacher_record_scientific_sha256=sample.teacher_record_scientific_sha256,
                descriptor_record=sample.descriptor_record,
            )
            for sample in selected
        ),
        teacher_cache_scientific_sha256=fixture.teacher_cache_scientific_sha256,
        teacher_cache_sample_coverage_sha256=fixture.teacher_cache_sample_coverage_sha256,
        descriptor_collection_scientific_sha256=(
            fixture.descriptor_collection_scientific_sha256
        ),
        split_scientific_sha256=fixture.split_scientific_sha256,
        checkpoint_sha256=fixture.checkpoint_sha256,
        execution_profile_sha256=fixture.execution_profile_sha256,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class HermeticContributionLayout:
    """Disk layout consumed by CLI / from-roots dry-run helpers."""

    root: Path
    teacher_cache_root: Path
    teacher_cache_manifest: Path
    descriptor_root: Path
    descriptor_manifest: Path
    mvtec_root: Path


def prepare_hermetic_contribution_inputs(
    tmp_path: Path,
    *,
    fixture: ContributionTargetFixture | None = None,
) -> HermeticContributionLayout:
    """Materialize a temporary teacher-cache + descriptor layout for CLI tests.

    Every artifact is marked ``artifact_kind == "test_fixture"``. Official
    production mode must refuse this layout.
    """

    fixture = fixture or build_contribution_target_fixture()
    root = Path(tmp_path) / "hermetic_contribution_inputs"
    if root.exists():
        raise AssertionError(f"hermetic contribution input root already exists: {root}")
    teacher_root = root / "teacher_cache"
    descriptor_root = root / "descriptor_collection"
    mvtec_root = root / "mvtec_source_stub"
    teacher_samples_dir = teacher_root / "samples"
    descriptor_dir = descriptor_root / "descriptors"
    teacher_samples_dir.mkdir(parents=True, exist_ok=False)
    descriptor_dir.mkdir(parents=True, exist_ok=False)
    mvtec_root.mkdir(parents=True, exist_ok=False)
    (mvtec_root / "README").write_text(
        "hermetic mvtec stub — never a real target-domain path\n",
        encoding="utf-8",
    )

    teacher_rows: list[dict[str, Any]] = []
    descriptor_rows: list[dict[str, Any]] = []
    for sample in fixture.samples:
        teacher_relative = f"samples/{sample.stable_sample_id}.pt"
        teacher_path = teacher_root / teacher_relative
        torch.save(
            {
                "scientific_record": dict(sample.teacher_record),
                "maps_by_depth": {
                    int(depth): {
                        int(layer): tensor.detach().cpu().clone()
                        for layer, tensor in layer_maps.items()
                    }
                    for depth, layer_maps in sample.maps_by_depth.items()
                },
                "mask": sample.mask.detach().cpu().clone(),
                "full_depth_map": sample.full_depth_map.detach().cpu().clone(),
            },
            teacher_path,
        )
        teacher_rows.append(
            {
                "stable_sample_id": sample.stable_sample_id,
                "membership": sample.membership,
                "relative_path": teacher_relative,
                "record_scientific_sha256": sample.teacher_record_scientific_sha256,
                "record_file_sha256": _sha256_file(teacher_path),
            }
        )

        descriptor_relative = f"descriptors/{sample.stable_sample_id}.pt"
        descriptor_path = descriptor_root / descriptor_relative
        torch.save(
            {"scientific_record": dict(sample.descriptor_record)},
            descriptor_path,
        )
        descriptor_rows.append(
            {
                "stable_sample_id": sample.stable_sample_id,
                "relative_record_path": descriptor_relative,
                "descriptor_record_scientific_sha256": sample.descriptor_record[
                    "descriptor_record_scientific_sha256"
                ],
                "descriptor_record_file_sha256": _sha256_file(descriptor_path),
            }
        )

    teacher_manifest = {
        "status": "passed",
        "artifact_kind": fixture.artifact_kind,
        "cache_scientific_sha256": fixture.teacher_cache_scientific_sha256,
        "sample_coverage_sha256": fixture.teacher_cache_sample_coverage_sha256,
        "split_scientific_sha256": fixture.split_scientific_sha256,
        "checkpoint_sha256": fixture.checkpoint_sha256,
        "execution_profile_sha256": fixture.execution_profile_sha256,
        "candidate_layers": list(fixture.candidate_layers),
        "prediction_depths": list(fixture.prediction_depths),
        "samples": teacher_rows,
    }
    descriptor_manifest = {
        "status": "passed",
        "artifact_kind": fixture.artifact_kind,
        "descriptor_collection_scientific_sha256": (
            fixture.descriptor_collection_scientific_sha256
        ),
        "candidate_layers": list(fixture.candidate_layers),
        "prediction_depths": list(fixture.prediction_depths),
        "samples": descriptor_rows,
    }
    teacher_manifest_path = teacher_root / "final_manifest.json"
    descriptor_manifest_path = descriptor_root / "final_manifest.json"
    teacher_manifest_path.write_text(
        json.dumps(teacher_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    descriptor_manifest_path.write_text(
        json.dumps(descriptor_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return HermeticContributionLayout(
        root=root,
        teacher_cache_root=teacher_root,
        teacher_cache_manifest=teacher_manifest_path,
        descriptor_root=descriptor_root,
        descriptor_manifest=descriptor_manifest_path,
        mvtec_root=mvtec_root,
    )

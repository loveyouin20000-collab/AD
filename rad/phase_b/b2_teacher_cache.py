"""Fail-closed teacher-cache config, planning, hashing, and Option A persistence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, NoReturn

from rad.runtime.execution_profile import (
    is_controlled_execution_profile_attestation,
)

# Production call-boundary names for monkeypatch tests. Bound lazily so Task 1
# config/provenance imports remain pre-bootstrap safe (no early torch import).
sum_preserving_fusion: Any = None
LayerDescriptorExtractor: Any = None
compute_exit_signals: Any = None

_MEMBERSHIPS = ("training", "calibration", "evaluation")
_EXPECTED_PROFILE_SHA256 = (
    "7af8dba39633743da0380fef9710940cded655f68c9efa8f84f5a52aeddb3c8d"
)
_EXPECTED_SPLIT_SHA256 = (
    "91570da1fed6d7859d407196b10403581832ae0ff677a1ea7657ca76b91471f0"
)
_EXPECTED_CHECKPOINT_SHA256 = (
    "97bd461163efb96e36cddb1c3adf677e4c4fc2daabb2521021689f30e799b4f4"
)
_EXPECTED_B1_COMMIT = "3a751b2784a50eb0a08ed49e1db2df0b53608ccc"
_EXPECTED_B2_COMMIT = "18bac047227754c975b23b46842458a5b41d5e2a"
_EXPECTED_CONFIG_CANONICAL_SHA256 = (
    "8205e8ffb97f24b7c2fe21e8c7be558427edbcf4699bfa5a9b06c53fdb425089"
)
_EXPECTED_DESCRIPTOR_IMPLEMENTATION_SHA256 = (
    "6846ad263d342649a0383c4f762f7820053428bf74a05ece8c02e1dcc641b615"
)
_DESCRIPTOR_SOURCE_TENSOR_KIND = "causal_anomaly_maps"
_DESCRIPTOR_EXTRACTOR_TOP_K_RATIO = 0.1
_SAMPLE_FIELDS = frozenset(
    {
        "stable_sample_id",
        "category",
        "image_label",
        "anomaly_type",
        "image_identity",
        "mask_identity",
        "membership",
    }
)
_FORBIDDEN_IDENTITY_COMPONENTS = frozenset(
    {"tests", "fixtures", "examples", "synthetic", "visa", "target"}
)
_RECORD_HASH_SCHEMA_VERSION = 1
_SCIENTIFIC_RECORD_FIELDS_V1: tuple[str, ...] = (
    "record_schema_version",
    "record_hash_schema_version",
    "stable_sample_id",
    "membership",
    "category",
    "image_label",
    "anomaly_type",
    "image_identity",
    "mask_identity",
    "candidate_layers",
    "prediction_depths",
    "causal_map_lattice",
    "cache_tensor_contract_version",
    "tensors",
    "descriptor_contract_version",
    "descriptor_feature_names",
    "descriptor_source_tensor_kind",
    "descriptor_implementation_sha256",
    "extractor_configuration_sha256",
    "split_scientific_hash_version",
    "split_scientific_sha256",
    "checkpoint_sha256",
    "execution_profile_name",
    "execution_profile_sha256",
)
_KNOWN_EXCLUDED_RECORD_FIELDS = frozenset(
    {
        "runtime_attestation_sha256",
        "generation_commit",
        "generation_branch",
        "worktree_clean",
        "worktree_status",
        "machine_hostname",
        "environment",
        "run_id",
        "output_path",
        "absolute_image_path",
        "absolute_mask_path",
        "checkpoint_path",
        "timestamp",
        "record_file_sha256",
    }
)


class TeacherCacheError(RuntimeError):
    """A teacher-cache contract failure carrying a stable error code."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise TeacherCacheError(code, detail)


@dataclass(frozen=True)
class TeacherCacheConfig:
    schema_version: int
    configuration_id: str
    candidate_layers: tuple[int, ...]
    prediction_depths: tuple[int, ...]
    split_scientific_hash_version: int
    split_scientific_sha256: str
    membership_counts: Mapping[str, int]
    total_selected_samples: int
    checkpoint_path: Path
    checkpoint_sha256: str
    execution_profile_name: str
    execution_profile_sha256: str
    b1_base_tag: str
    b1_base_commit: str
    b2_base_tag: str
    b2_base_commit: str
    cache_tensor_contract_version: int
    descriptor_contract_version: int
    descriptor_implementation_sha256: str
    descriptor_source_tensor_kind: str
    record_hash_schema_version: int
    source_dataset: str
    forbidden_target_dataset: str


@dataclass(frozen=True)
class PlannedSample:
    stable_sample_id: str
    membership: str
    category: str
    image_label: int
    anomaly_type: str
    image_identity: str
    mask_identity: str | None


@dataclass(frozen=True)
class OuterProvenance:
    execution_profile_sha256: str
    runtime_attestation: Mapping[str, Any]
    runtime_attestation_sha256: str
    split_scientific_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    b2_tag_commit: str
    head_commit: str
    head_is_descendant: bool
    worktree_clean: bool
    forbidden_target_access_count: int


@dataclass(frozen=True, order=True)
class MapIdentity:
    """Explicit identity for a checkpoint-conditioned candidate-layer map."""

    checkpoint_depth: int
    candidate_layer_id: int


@dataclass(frozen=True)
class CacheContract:
    """Tensor contract required to validate one teacher output."""

    candidate_layers: tuple[int, ...]
    prediction_depths: tuple[int, ...]
    backbone_depth: int
    expected_sample_ids: frozenset[str]
    map_shape: tuple[int, ...]
    map_dimension_semantics: tuple[str, ...]
    production_mode: bool = False

    def __post_init__(self) -> None:
        sequences = (self.candidate_layers, self.prediction_depths)
        if (
            any(not sequence or tuple(sorted(set(sequence))) != sequence for sequence in sequences)
            or not isinstance(self.backbone_depth, int)
            or self.backbone_depth < 1
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
                or value > self.backbone_depth
                for sequence in sequences
                for value in sequence
            )
            or len(self.map_shape) != 4
            or any(not isinstance(size, int) or size < 1 for size in self.map_shape)
            or len(self.map_dimension_semantics) != len(self.map_shape)
        ):
            _fail(
                "B2_CACHE_TENSOR_CONTRACT_INVALID",
                "layers, depths, shape, or dimension semantics are invalid",
            )
        if not self.expected_sample_ids:
            _fail(
                "B2_CACHE_EXPECTED_SAMPLE_MISSING",
                "the exact expected sample set is empty",
            )


@dataclass(frozen=True)
class TeacherOutput:
    """Untrusted output emitted by a teacher forward."""

    sample_id: str
    image_label: int
    anomalous_mask: Any
    maps: Mapping[MapIdentity, Any]
    map_dimension_semantics: Mapping[MapIdentity, tuple[str, ...]]
    descriptor_source_identities: frozenset[MapIdentity]
    artifact_kind: str = "production"


@dataclass(frozen=True)
class ValidatedTeacherOutput:
    """Teacher output after exact identity and tensor validation."""

    sample_id: str
    image_label: int
    anomalous_mask: Any
    maps: Mapping[MapIdentity, Any]
    map_dimension_semantics: Mapping[MapIdentity, tuple[str, ...]]
    candidate_layers: tuple[int, ...]
    prediction_depths: tuple[int, ...]


def expected_lattice(
    candidate_layers: tuple[int, ...],
    prediction_depths: tuple[int, ...],
) -> frozenset[MapIdentity]:
    """Return the configuration-derived causal-map identity lattice."""

    return frozenset(
        MapIdentity(depth, layer)
        for depth in prediction_depths
        for layer in candidate_layers
        if layer <= depth
    )


def _bind_production_tensor_apis() -> Any:
    """Lazily bind production tensor APIs while preserving monkeypatch hooks."""

    global sum_preserving_fusion, LayerDescriptorExtractor, compute_exit_signals
    import torch

    if sum_preserving_fusion is None:
        from rad.models.dlcm import sum_preserving_fusion as _fusion

        sum_preserving_fusion = _fusion
    if LayerDescriptorExtractor is None:
        from rad.models.descriptors import (
            LayerDescriptorExtractor as _extractor,
        )

        LayerDescriptorExtractor = _extractor
    if compute_exit_signals is None:
        from rad.inference.adaptive_engine import (
            compute_exit_signals as _exit_signals,
        )

        compute_exit_signals = _exit_signals
    return torch


def _validate_float32_tensor(
    tensor: Any,
    *,
    expected_shape: tuple[int, ...],
    role: str,
) -> None:
    torch = _bind_production_tensor_apis()
    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_shape:
        _fail("B2_CACHE_MAP_SHAPE_INVALID", f"{role} has the wrong shape")
    if tensor.dtype != torch.float32:
        _fail("B2_CACHE_TENSOR_DTYPE_INVALID", f"{role} must be float32")
    if not bool(torch.isfinite(tensor).all()):
        _fail("B2_CACHE_TENSOR_NONFINITE", f"{role} contains NaN or Inf")


def validate_teacher_output(
    output: TeacherOutput,
    contract: CacheContract,
) -> ValidatedTeacherOutput:
    """Fail closed unless a teacher output exactly satisfies the cache contract."""

    if contract.production_mode and output.artifact_kind == "test_fixture":
        _fail(
            "B2_CACHE_TEST_FIXTURE_FORBIDDEN",
            "production mode cannot accept a test-fixture teacher",
        )
    if output.sample_id not in contract.expected_sample_ids:
        _fail("B2_CACHE_UNEXPECTED_SAMPLE", f"sample {output.sample_id!r} is not planned")
    required = expected_lattice(contract.candidate_layers, contract.prediction_depths)
    actual = frozenset(output.maps)
    semantics_set = frozenset(output.map_dimension_semantics)
    if (
        actual != required
        or semantics_set != required
        or output.descriptor_source_identities != required
    ):
        _fail(
            "B2_CACHE_MAP_LATTICE_MISMATCH",
            "map and descriptor-source identities must exactly match the lattice",
        )
    for identity in sorted(required):
        tensor = output.maps[identity]
        _validate_float32_tensor(
            tensor,
            expected_shape=contract.map_shape,
            role=f"map {identity}",
        )
        if (
            tuple(output.map_dimension_semantics[identity])
            != contract.map_dimension_semantics
        ):
            _fail(
                "B2_CACHE_DIMENSION_SEMANTICS_INVALID",
                f"map {identity} has incorrect dimension semantics",
            )
    if output.image_label not in (0, 1):
        _fail("B2_CACHE_TENSOR_CONTRACT_INVALID", "image label must be binary")
    if output.image_label == 1 and output.anomalous_mask is None:
        _fail(
            "B2_CACHE_ANOMALOUS_MASK_MISSING",
            f"anomalous sample {output.sample_id} lacks a mask",
        )
    if output.anomalous_mask is not None:
        _validate_float32_tensor(
            output.anomalous_mask,
            expected_shape=contract.map_shape,
            role="anomalous mask",
        )
    return ValidatedTeacherOutput(
        sample_id=output.sample_id,
        image_label=output.image_label,
        anomalous_mask=output.anomalous_mask,
        maps=MappingProxyType(dict(output.maps)),
        map_dimension_semantics=MappingProxyType(
            dict(output.map_dimension_semantics)
        ),
        candidate_layers=contract.candidate_layers,
        prediction_depths=contract.prediction_depths,
    )


def _maps_at_depth(
    validated: ValidatedTeacherOutput,
    depth: int,
) -> tuple[Any, Any]:
    torch = _bind_production_tensor_apis()
    identities = tuple(
        MapIdentity(depth, layer)
        for layer in validated.candidate_layers
        if layer <= depth
    )
    maps = torch.stack([validated.maps[identity] for identity in identities], dim=1)
    valid_mask = torch.ones(
        maps.shape[:2],
        dtype=torch.bool,
        device=maps.device,
    )
    return maps, valid_mask


def build_cumulative_maps(
    validated: ValidatedTeacherOutput,
) -> Mapping[int, Any]:
    """Build sum-scale cumulative maps through the production fusion function."""

    _bind_production_tensor_apis()
    cumulative: dict[int, Any] = {}
    for depth in validated.prediction_depths:
        maps, valid_mask = _maps_at_depth(validated, depth)
        weights = valid_mask.to(maps.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True)
        fused = sum_preserving_fusion(maps, weights, valid_mask)
        _validate_float32_tensor(
            fused,
            expected_shape=tuple(maps.shape[0:1] + maps.shape[2:]),
            role=f"cumulative map at depth {depth}",
        )
        cumulative[depth] = fused
    return MappingProxyType(cumulative)


def reconstruct_descriptors(
    validated: ValidatedTeacherOutput,
) -> Mapping[int, Any]:
    """Reconstruct descriptors through the authoritative production extractor."""

    _bind_production_tensor_apis()
    extractor = LayerDescriptorExtractor()
    descriptors: dict[int, Any] = {}
    for depth in validated.prediction_depths:
        maps, valid_mask = _maps_at_depth(validated, depth)
        descriptors[depth] = extractor(maps.squeeze(2), valid_mask)
    return MappingProxyType(descriptors)


def compute_final_image_score(full_depth_map: Any) -> Any:
    """Compute the image score from the sum-preserving full-depth map."""

    torch = _bind_production_tensor_apis()
    if (
        not isinstance(full_depth_map, torch.Tensor)
        or full_depth_map.ndim != 4
        or full_depth_map.shape[1] != 1
    ):
        _fail(
            "B2_CACHE_MAP_SHAPE_INVALID",
            "full-depth map must have shape [batch, 1, height, width]",
        )
    _validate_float32_tensor(
        full_depth_map,
        expected_shape=tuple(full_depth_map.shape),
        role="full-depth map",
    )
    signals = compute_exit_signals(full_depth_map, None)
    score = torch.tensor(
        [signals.image_score],
        dtype=torch.float32,
        device=full_depth_map.device,
    )
    if not bool(torch.isfinite(score).all()):
        _fail("B2_CACHE_TENSOR_NONFINITE", "image score contains NaN or Inf")
    return score


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("B2_CACHE_CONFIG_INVALID", f"{field} must be an object")
    return value


def _require_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        _fail("B2_CACHE_CONFIG_INVALID", f"{field} must be an integer")
    return value


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("B2_CACHE_CONFIG_INVALID", f"{field} must be a non-empty string")
    return value


def _int_tuple(value: Any, field: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        _fail("B2_CACHE_CONFIG_INVALID", f"{field} must be a non-empty integer list")
    result = tuple(value)
    if tuple(sorted(set(result))) != result:
        _fail("B2_CACHE_CONFIG_INVALID", f"{field} must be sorted and unique")
    return result


def load_teacher_cache_config(path: Path) -> TeacherCacheConfig:
    """Load and validate the fixed B2-02A teacher-cache configuration."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail("B2_CACHE_CONFIG_INVALID", f"cannot load configuration: {exc}")
    root = _mapping(raw, "configuration")
    canonical_sha256 = hashlib.sha256(
        json.dumps(
            root,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    if canonical_sha256 != _EXPECTED_CONFIG_CANONICAL_SHA256:
        _fail(
            "B2_CACHE_CONFIG_MISMATCH",
            "configuration differs from the complete approved Gate C contract",
        )
    split = _mapping(root.get("split"), "split")
    checkpoint = _mapping(root.get("checkpoint"), "checkpoint")
    profile = _mapping(root.get("execution_profile"), "execution_profile")
    b1 = _mapping(root.get("b1_base"), "b1_base")
    b2 = _mapping(root.get("b2_base"), "b2_base")
    contracts = _mapping(root.get("contracts"), "contracts")
    counts_raw = _mapping(split.get("membership_counts"), "membership_counts")
    if set(counts_raw) != set(_MEMBERSHIPS) or any(
        not isinstance(counts_raw[name], int) or counts_raw[name] <= 0
        for name in _MEMBERSHIPS
    ):
        _fail("B2_CACHE_CONFIG_INVALID", "membership counts are invalid")
    counts = {name: counts_raw[name] for name in _MEMBERSHIPS}
    candidate_layers = _int_tuple(root.get("candidate_layers"), "candidate_layers")
    prediction_depths = _int_tuple(root.get("prediction_depths"), "prediction_depths")
    config = TeacherCacheConfig(
        schema_version=_require_int(root.get("schema_version"), "schema_version"),
        configuration_id=_require_str(root.get("configuration_id"), "configuration_id"),
        candidate_layers=candidate_layers,
        prediction_depths=prediction_depths,
        split_scientific_hash_version=_require_int(
            split.get("scientific_hash_version"), "scientific_hash_version"
        ),
        split_scientific_sha256=_require_str(
            split.get("scientific_sha256"), "scientific_sha256"
        ),
        membership_counts=MappingProxyType(counts),
        total_selected_samples=_require_int(
            split.get("total_selected_samples"), "total_selected_samples"
        ),
        checkpoint_path=Path(_require_str(checkpoint.get("path"), "checkpoint.path")),
        checkpoint_sha256=_require_str(checkpoint.get("sha256"), "checkpoint.sha256"),
        execution_profile_name=_require_str(profile.get("name"), "execution_profile.name"),
        execution_profile_sha256=_require_str(
            profile.get("sha256"), "execution_profile.sha256"
        ),
        b1_base_tag=_require_str(b1.get("tag"), "b1_base.tag"),
        b1_base_commit=_require_str(b1.get("commit"), "b1_base.commit"),
        b2_base_tag=_require_str(b2.get("tag"), "b2_base.tag"),
        b2_base_commit=_require_str(b2.get("commit"), "b2_base.commit"),
        cache_tensor_contract_version=_require_int(
            contracts.get("cache_tensor_contract_version"),
            "cache_tensor_contract_version",
        ),
        descriptor_contract_version=_require_int(
            contracts.get("descriptor_contract_version"),
            "descriptor_contract_version",
        ),
        descriptor_implementation_sha256=_require_str(
            contracts.get("descriptor_implementation_sha256"),
            "descriptor_implementation_sha256",
        ),
        descriptor_source_tensor_kind=_require_str(
            contracts.get("descriptor_source_tensor_kind"),
            "descriptor_source_tensor_kind",
        ),
        record_hash_schema_version=_require_int(
            contracts.get("record_hash_schema_version"),
            "record_hash_schema_version",
        ),
        source_dataset=_require_str(root.get("source_dataset"), "source_dataset"),
        forbidden_target_dataset=_require_str(
            root.get("forbidden_target_dataset"), "forbidden_target_dataset"
        ),
    )
    if (
        config.schema_version != 1
        or config.configuration_id != "b2_teacher_cache_gate_c"
        or config.split_scientific_hash_version != 2
        or config.split_scientific_sha256 != _EXPECTED_SPLIT_SHA256
        or config.checkpoint_sha256 != _EXPECTED_CHECKPOINT_SHA256
        or config.execution_profile_name != "frozen_deterministic_math"
        or config.execution_profile_sha256 != _EXPECTED_PROFILE_SHA256
        or config.b1_base_tag != "b1-strict-independent-v1"
        or config.b1_base_commit != _EXPECTED_B1_COMMIT
        or config.b2_base_tag != "b2-tiny-split-v1"
        or config.b2_base_commit != _EXPECTED_B2_COMMIT
        or config.membership_counts
        != {"training": 16, "calibration": 8, "evaluation": 8}
        or config.total_selected_samples != 32
        or config.source_dataset != "mvtec"
        or config.forbidden_target_dataset != "visa"
        or config.descriptor_implementation_sha256
        != _EXPECTED_DESCRIPTOR_IMPLEMENTATION_SHA256
        or config.descriptor_source_tensor_kind != _DESCRIPTOR_SOURCE_TENSOR_KIND
        or any(
            version != 1
            for version in (
                config.cache_tensor_contract_version,
                config.descriptor_contract_version,
                config.record_hash_schema_version,
            )
        )
    ):
        _fail("B2_CACHE_CONFIG_MISMATCH", "configuration differs from approved values")
    if any(layer > candidate_layers[-1] for layer in prediction_depths):
        _fail("B2_CACHE_CONFIG_INVALID", "prediction depth exceeds backbone depth")
    return config


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _split_scientific_sha256(manifest: Mapping[str, Any]) -> str:
    from rad.phase_b.b2_tiny_split import canonical_scientific_hash_v2

    return canonical_scientific_hash_v2(manifest)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identity_parts(value: Any, *, stable_id: str, field: str) -> tuple[str, ...]:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or value.startswith("/")
        or value.endswith("/")
        or "//" in value
    ):
        _fail(
            "B2_CACHE_SAMPLE_IDENTITY_INVALID",
            f"sample {stable_id} has invalid {field}",
        )
    raw_parts = tuple(value.split("/"))
    if (
        any(part in {"", ".", ".."} for part in raw_parts)
        or _FORBIDDEN_IDENTITY_COMPONENTS.intersection(
            part.lower() for part in raw_parts
        )
        or PurePosixPath(value).is_absolute()
    ):
        _fail(
            "B2_CACHE_SAMPLE_IDENTITY_INVALID",
            f"sample {stable_id} has forbidden {field}",
        )
    return raw_parts


def _validate_sample_identity(row: Mapping[str, Any], stable_id: str) -> None:
    category = row["category"]
    anomaly_type = row["anomaly_type"]
    label = row["image_label"]
    if (
        not isinstance(category, str)
        or not category
        or not isinstance(anomaly_type, str)
        or not anomaly_type
        or "/" in category
        or "/" in anomaly_type
        or "\\" in category
        or "\\" in anomaly_type
    ):
        _fail(
            "B2_CACHE_SAMPLE_IDENTITY_INVALID",
            f"sample {stable_id} category or anomaly type is invalid",
        )
    image_parts = _identity_parts(
        row["image_identity"], stable_id=stable_id, field="image_identity"
    )
    if (
        len(image_parts) != 4
        or image_parts[0] != category
        or image_parts[1] != "test"
        or image_parts[2] != anomaly_type
    ):
        _fail(
            "B2_CACHE_SAMPLE_IDENTITY_INVALID",
            f"sample {stable_id} image schema is inconsistent",
        )
    mask_identity = row["mask_identity"]
    if label == 1:
        mask_parts = _identity_parts(
            mask_identity, stable_id=stable_id, field="mask_identity"
        )
        image_path = PurePosixPath(row["image_identity"])
        mask_path = PurePosixPath(mask_identity)
        if (
            len(mask_parts) != 4
            or mask_parts[0] != category
            or mask_parts[1] != "ground_truth"
            or mask_parts[2] != anomaly_type
            or mask_path.name != f"{image_path.stem}_mask{image_path.suffix}"
        ):
            _fail(
                "B2_CACHE_SAMPLE_IDENTITY_INVALID",
                f"sample {stable_id} mask schema is inconsistent",
            )
    expected_stable_id = _canonical_sha256(
        {
            "dataset": "mvtec",
            "category": category,
            "source_split": "test",
            "anomaly_type": anomaly_type,
            "image_identity": row["image_identity"],
        }
    )
    if stable_id != expected_stable_id:
        _fail(
            "B2_CACHE_SELECTED_ID_DRIFT",
            f"sample {stable_id} does not match its scientific identity",
        )


def validate_split_manifest(
    manifest: Mapping[str, Any], config: TeacherCacheConfig
) -> None:
    """Validate the accepted V2 split and its exact selected sample set."""

    contract = manifest.get("scientific_hash_contract")
    if not isinstance(contract, Mapping) or contract.get("active_version") != 2:
        _fail("B2_CACHE_SPLIT_V2_REQUIRED", "current split identity must be V2")
    claimed_hash = contract.get("canonical_scientific_hash_v2")
    if claimed_hash != config.split_scientific_sha256:
        _fail("B2_CACHE_SPLIT_HASH_MISMATCH", "split V2 hash is not approved")
    if (
        manifest.get("transfer_direction") != "mvtec_to_visa"
        or manifest.get("forbidden_target_dataset") != config.forbidden_target_dataset
    ):
        _fail("B2_CACHE_TARGET_ACCESS_FORBIDDEN", "split domain identity drifted")
    source_audit = manifest.get("source_only_audit")
    if (
        not isinstance(source_audit, Mapping)
        or source_audit.get("passed") is not True
        or source_audit.get("forbidden_target_access_count") != 0
    ):
        _fail("B2_CACHE_TARGET_ACCESS_FORBIDDEN", "target access was not zero")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(_MEMBERSHIPS):
        _fail("B2_CACHE_SPLIT_COUNT_MISMATCH", "split memberships are incomplete")

    seen: set[str] = set()
    for membership in _MEMBERSHIPS:
        rows = splits[membership]
        if (
            not isinstance(rows, list | tuple)
            or len(rows) != config.membership_counts[membership]
        ):
            _fail(
                "B2_CACHE_SPLIT_COUNT_MISMATCH",
                f"{membership} count does not match configuration",
            )
        for row in rows:
            if not isinstance(row, Mapping):
                _fail("B2_CACHE_SAMPLE_SCHEMA_INVALID", "sample row must be an object")
            if set(row) != _SAMPLE_FIELDS:
                _fail(
                    "B2_CACHE_SAMPLE_SCHEMA_INVALID",
                    "sample row fields do not match the exact schema",
                )
            stable_id_raw = row.get("stable_sample_id")
            if not _is_sha256(stable_id_raw):
                _fail("B2_CACHE_SAMPLE_SCHEMA_INVALID", "stable sample ID is invalid")
            stable_id = str(stable_id_raw)
            if stable_id in seen:
                _fail("B2_CACHE_DUPLICATE_SAMPLE", f"duplicate sample {stable_id}")
            seen.add(stable_id)
            if row.get("membership") != membership:
                _fail(
                    "B2_CACHE_MEMBERSHIP_MISMATCH",
                    f"sample {stable_id} has the wrong membership",
                )
            label = row.get("image_label")
            mask_identity = row.get("mask_identity")
            if label not in (0, 1):
                _fail(
                    "B2_CACHE_SAMPLE_SCHEMA_INVALID",
                    f"sample {stable_id} has invalid label",
                )
            if label == 1 and not isinstance(mask_identity, str):
                _fail(
                    "B2_CACHE_ANOMALOUS_MASK_MISSING",
                    f"sample {stable_id} lacks an anomalous mask identity",
                )
            if label == 0 and mask_identity is not None:
                _fail(
                    "B2_CACHE_SAMPLE_IDENTITY_INVALID",
                    f"normal sample {stable_id} has a mask",
                )
            _validate_sample_identity(row, stable_id)
    if len(seen) != config.total_selected_samples:
        _fail("B2_CACHE_SPLIT_COUNT_MISMATCH", "selected sample total is not exact")
    if _split_scientific_sha256(manifest) != config.split_scientific_sha256:
        _fail("B2_CACHE_SELECTED_ID_DRIFT", "selected IDs drifted from split V2")


def build_generation_plan(
    manifest: Mapping[str, Any], config: TeacherCacheConfig
) -> tuple[PlannedSample, ...]:
    """Build the immutable plan while preserving accepted split sample order."""

    validate_split_manifest(manifest, config)
    splits = manifest["splits"]
    return tuple(
        PlannedSample(
            stable_sample_id=row["stable_sample_id"],
            membership=membership,
            category=row["category"],
            image_label=row["image_label"],
            anomaly_type=row["anomaly_type"],
            image_identity=row["image_identity"],
            mask_identity=row["mask_identity"],
        )
        for membership in _MEMBERSHIPS
        for row in splits[membership]
    )


def _thaw_for_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_for_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_thaw_for_json(item) for item in value]
    return value


def validate_outer_provenance(
    *,
    config: TeacherCacheConfig,
    bootstrap_validated: bool,
    execution_profile_sha256: str,
    runtime_attestation: Any,
    split_manifest: Mapping[str, Any] | None,
    checkpoint_path: Path | None,
    b2_tag_commit: str | None,
    head_commit: str,
    head_is_descendant: bool,
    worktree_clean: bool,
    forbidden_target_access_count: int,
) -> OuterProvenance:
    """Validate fail-closed runtime, input, repository, and environment evidence."""

    if bootstrap_validated is not True:
        _fail("B2_CACHE_BOOTSTRAP_REQUIRED", "validated launcher bootstrap is absent")
    if execution_profile_sha256 != config.execution_profile_sha256:
        _fail("B2_CACHE_PROFILE_HASH_MISMATCH", "execution profile hash drifted")
    if not is_controlled_execution_profile_attestation(runtime_attestation):
        _fail("B2_CACHE_RUNTIME_ATTESTATION_REQUIRED", "attestation is absent")
    canonical = runtime_attestation.canonical_attestation()
    canonical_sha256 = hashlib.sha256(
        json.dumps(
            _thaw_for_json(canonical),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    provenance = runtime_attestation.artifact_provenance()
    expected_provenance = {
        "execution_profile_name": config.execution_profile_name,
        "execution_profile_sha256": config.execution_profile_sha256,
        "runtime_attestation_sha256": runtime_attestation.attestation_sha256,
    }
    profile = canonical.get("profile")
    if (
        not isinstance(canonical, Mapping)
        or not isinstance(profile, Mapping)
        or profile.get("profile_id") != config.execution_profile_name
        or profile.get("runtime_sha256") != config.execution_profile_sha256
        or canonical_sha256 != runtime_attestation.attestation_sha256
        or dict(provenance) != expected_provenance
    ):
        _fail("B2_CACHE_RUNTIME_ATTESTATION_INVALID", "attestation evidence drifted")
    if split_manifest is None:
        _fail("B2_CACHE_SPLIT_REQUIRED", "accepted split manifest is absent")
    validate_split_manifest(split_manifest, config)
    if checkpoint_path is None or not checkpoint_path.is_file():
        _fail("B2_CACHE_CHECKPOINT_MISSING", "checkpoint file is absent")
    if checkpoint_path.resolve() != config.checkpoint_path.resolve():
        _fail("B2_CACHE_CHECKPOINT_MISSING", "checkpoint path is not approved")
    # Byte-hash only: do not import the VisualAD production teacher loader.
    checkpoint_sha256 = validate_checkpoint_bytes(
        checkpoint_path, config.checkpoint_sha256
    )
    if b2_tag_commit is None:
        _fail("B2_CACHE_B2_TAG_UNRESOLVED", "B2 base tag did not resolve")
    if b2_tag_commit != config.b2_base_commit:
        _fail("B2_CACHE_B2_TAG_MOVED", "B2 base tag moved")
    if not _is_git_commit(head_commit):
        _fail("B2_CACHE_HEAD_INVALID", "HEAD commit is invalid")
    if head_is_descendant is not True:
        _fail("B2_CACHE_HEAD_NOT_DESCENDANT", "HEAD is not a B2-base descendant")
    if worktree_clean is not True:
        _fail("B2_CACHE_WORKTREE_DIRTY", "official worktree is dirty")
    if forbidden_target_access_count != 0:
        _fail("B2_CACHE_TARGET_ACCESS_FORBIDDEN", "target-domain access was observed")
    return OuterProvenance(
        execution_profile_sha256=execution_profile_sha256,
        runtime_attestation=canonical,
        runtime_attestation_sha256=runtime_attestation.attestation_sha256,
        split_scientific_sha256=config.split_scientific_sha256,
        checkpoint_path=checkpoint_path.resolve(),
        checkpoint_sha256=checkpoint_sha256,
        b2_tag_commit=b2_tag_commit,
        head_commit=head_commit,
        head_is_descendant=True,
        worktree_clean=True,
        forbidden_target_access_count=0,
    )


def validate_checkpoint_bytes(path: Path, expected_sha256: str) -> str:
    """Validate checkpoint identity by file bytes only; never load a model."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        _fail("B2_CACHE_CHECKPOINT_MISSING", "checkpoint file is absent")
    if not _is_sha256(expected_sha256):
        _fail("B2_CACHE_CHECKPOINT_HASH_MISMATCH", "checkpoint hash is invalid")
    actual = _sha256_file(checkpoint)
    if actual != expected_sha256:
        _fail("B2_CACHE_CHECKPOINT_HASH_MISMATCH", "checkpoint bytes drifted")
    return actual


def build_intended_manifest_metadata(
    *,
    config: TeacherCacheConfig,
    plan: Sequence[PlannedSample],
    provenance: OuterProvenance,
) -> Mapping[str, Any]:
    """Construct in-memory intended run metadata for dry-run planning."""

    if len(plan) != config.total_selected_samples:
        _fail("B2_CACHE_SPLIT_COUNT_MISMATCH", "plan cardinality drifted")
    return MappingProxyType(
        {
            "status": "intended",
            "schema_version": 1,
            "configuration_id": config.configuration_id,
            "planned_samples": len(plan),
            "planned_stable_sample_ids": [row.stable_sample_id for row in plan],
            "membership_counts": {
                name: sum(1 for row in plan if row.membership == name)
                for name in _MEMBERSHIPS
            },
            "candidate_layers": list(config.candidate_layers),
            "prediction_depths": list(config.prediction_depths),
            "split_scientific_hash_version": config.split_scientific_hash_version,
            "split_scientific_sha256": config.split_scientific_sha256,
            "checkpoint_sha256": provenance.checkpoint_sha256,
            "execution_profile_name": config.execution_profile_name,
            "execution_profile_sha256": provenance.execution_profile_sha256,
            "runtime_attestation_sha256": provenance.runtime_attestation_sha256,
            "b2_tag_commit": provenance.b2_tag_commit,
            "head_commit": provenance.head_commit,
            "worktree_clean": provenance.worktree_clean,
            "cache_tensor_contract_version": config.cache_tensor_contract_version,
            "descriptor_contract_version": config.descriptor_contract_version,
            "descriptor_implementation_sha256": config.descriptor_implementation_sha256,
            "record_hash_schema_version": config.record_hash_schema_version,
        }
    )


def require_production_teacher(teacher: Any) -> Any:
    """Hard-reject test-fixture teachers in production mode."""

    kind = getattr(teacher, "artifact_kind", None)
    if kind == "test_fixture":
        _fail(
            "B2_CACHE_TEST_TEACHER_FORBIDDEN",
            "production mode cannot accept a test-fixture teacher",
        )
    if kind != "production":
        _fail(
            "B2_CACHE_TEST_TEACHER_FORBIDDEN",
            "production mode requires a production teacher",
        )
    return teacher


def descriptor_implementation_sha256(repo_root: Path) -> str:
    """SHA-256 of the exact tracked bytes of rad/models/descriptors.py."""

    path = Path(repo_root) / "rad" / "models" / "descriptors.py"
    try:
        payload = path.read_bytes()
    except OSError as exc:
        _fail(
            "B2_CACHE_DESCRIPTOR_CONTRACT_DRIFT",
            f"cannot read descriptors implementation: {exc}",
        )
    return hashlib.sha256(payload).hexdigest()


def _extractor_configuration_sha256(
    *,
    feature_names: tuple[str, ...],
    descriptor_contract_version: int,
) -> str:
    from rad.models.descriptors import LayerDescriptorExtractor

    return _canonical_sha256(
        {
            "descriptor_contract_version": descriptor_contract_version,
            "extractor_class": (
                f"{LayerDescriptorExtractor.__module__}."
                f"{LayerDescriptorExtractor.__qualname__}"
            ),
            "feature_names": list(feature_names),
            "top_k_ratio": _DESCRIPTOR_EXTRACTOR_TOP_K_RATIO,
        }
    )


def descriptor_contract(
    config: TeacherCacheConfig,
    repo_root: Path,
) -> Mapping[str, Any]:
    """Build the authoritative descriptor contract; fail closed on drift."""

    from rad.models.descriptors import LAYER_DESCRIPTOR_FEATURE_NAMES

    implementation_digest = descriptor_implementation_sha256(repo_root)
    feature_names = tuple(LAYER_DESCRIPTOR_FEATURE_NAMES)
    if (
        config.descriptor_contract_version != 1
        or config.descriptor_source_tensor_kind != _DESCRIPTOR_SOURCE_TENSOR_KIND
        or config.descriptor_implementation_sha256 != implementation_digest
        or implementation_digest != _EXPECTED_DESCRIPTOR_IMPLEMENTATION_SHA256
        or feature_names
        != tuple(
            (
                "margin_mean",
                "margin_std",
                "margin_max",
                "margin_topk",
                "background_contrast",
                "response_topk_mean",
                "response_max",
                "sparsity",
                "top_entropy",
                "global_entropy",
                "rank_spearman",
                "topk_overlap",
                "fused_map_change",
                "response_comp",
                "absolute_comp",
                "boundary_comp",
                "response_trend",
                "entropy_trend",
            )
        )
    ):
        _fail(
            "B2_CACHE_DESCRIPTOR_CONTRACT_DRIFT",
            "descriptor contract drifted from the approved Gate C pin",
        )
    contract = {
        "descriptor_contract_version": config.descriptor_contract_version,
        "feature_names": feature_names,
        "descriptor_source_tensor_kind": config.descriptor_source_tensor_kind,
        "descriptor_implementation_sha256": implementation_digest,
        "extractor_configuration_sha256": _extractor_configuration_sha256(
            feature_names=feature_names,
            descriptor_contract_version=config.descriptor_contract_version,
        ),
    }
    return MappingProxyType(contract)


def validate_descriptor_contract(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Fail closed unless a loaded descriptor contract matches the expected one."""

    if set(actual) != set(expected):
        _fail(
            "B2_CACHE_DESCRIPTOR_CONTRACT_DRIFT",
            "descriptor contract fields differ",
        )
    for key, value in expected.items():
        actual_value = actual[key]
        if key == "feature_names":
            if tuple(actual_value) != tuple(value):
                _fail(
                    "B2_CACHE_DESCRIPTOR_CONTRACT_DRIFT",
                    "descriptor feature order drifted",
                )
            continue
        if actual_value != value:
            _fail(
                "B2_CACHE_DESCRIPTOR_CONTRACT_DRIFT",
                f"descriptor contract field {key} drifted",
            )


def _u64_le(value: int) -> bytes:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _fail("B2_CACHE_TENSOR_UNSUPPORTED", "canonical length must be a uint64")
    return value.to_bytes(8, "little", signed=False)


def _length_delimited_bytes(payload: bytes) -> bytes:
    return _u64_le(len(payload)) + payload


def _length_delimited_text(value: str) -> bytes:
    if not isinstance(value, str):
        _fail("B2_CACHE_TENSOR_UNSUPPORTED", "canonical text must be a string")
    return _length_delimited_bytes(value.encode("utf-8"))


def _dtype_name(tensor: Any) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def _little_endian_contiguous_bytes(tensor: Any) -> bytes:
    import numpy as np

    array = tensor.detach().to("cpu").contiguous().numpy()
    little = np.asarray(array, dtype=array.dtype.newbyteorder("<"))
    if little.dtype.byteorder == ">":
        little = little.byteswap().newbyteorder("<")
    return np.ascontiguousarray(little).tobytes(order="C")


def _encode_canonical_tensor_payload(
    name: str,
    dtype_name: str,
    shape: tuple[int, ...],
    dimension_semantics: tuple[str, ...],
    raw_bytes: bytes,
) -> bytes:
    parts = bytearray()
    parts.extend(_length_delimited_text(name))
    parts.extend(_length_delimited_text(dtype_name))
    parts.extend(_u64_le(len(shape)))
    for size in shape:
        parts.extend(_u64_le(int(size)))
    parts.extend(_u64_le(len(dimension_semantics)))
    for item in dimension_semantics:
        parts.extend(_length_delimited_text(item))
    parts.extend(_length_delimited_bytes(raw_bytes))
    return bytes(parts)


def canonical_tensor_digest(
    name: str,
    tensor: Any,
    dimension_semantics: tuple[str, ...] | list[str],
) -> str:
    """Hash one tensor with length-delimited canonical scientific metadata."""

    torch = _bind_production_tensor_apis()
    if not isinstance(name, str) or not name:
        _fail("B2_CACHE_TENSOR_UNSUPPORTED", "logical tensor name is required")
    if not isinstance(tensor, torch.Tensor):
        _fail("B2_CACHE_TENSOR_UNSUPPORTED", "value is not a torch.Tensor")
    if getattr(tensor, "is_sparse", False) or tensor.layout != torch.strided:
        _fail("B2_CACHE_TENSOR_UNSUPPORTED", "sparse tensors are rejected")
    if bool(getattr(tensor, "is_quantized", False)):
        _fail("B2_CACHE_TENSOR_UNSUPPORTED", "quantized tensors are rejected")
    if getattr(tensor, "is_nested", False):
        _fail("B2_CACHE_TENSOR_UNSUPPORTED", "nested tensors are rejected")
    semantics = tuple(dimension_semantics)
    if len(semantics) != int(tensor.ndim) or any(
        not isinstance(item, str) or not item for item in semantics
    ):
        _fail(
            "B2_CACHE_DIMENSION_SEMANTICS_INVALID",
            "dimension semantics must match tensor rank",
        )
    if not bool(torch.isfinite(tensor.detach()).all()):
        _fail("B2_CACHE_TENSOR_NONFINITE", f"tensor {name!r} contains NaN or Inf")
    dtype_name = _dtype_name(tensor)
    shape = tuple(int(size) for size in tensor.shape)
    payload = _encode_canonical_tensor_payload(
        name,
        dtype_name,
        shape,
        semantics,
        _little_endian_contiguous_bytes(tensor),
    )
    return hashlib.sha256(payload).hexdigest()


def _canonicalize_tensor_meta(name: str, meta: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(meta, Mapping):
        _fail("B2_CACHE_RECORD_HASH_SCHEMA_INVALID", f"tensor {name} meta invalid")
    required = {
        "logical_name",
        "dtype",
        "shape",
        "dimension_semantics",
        "digest",
    }
    allowed = required | {"tensor"}
    if set(meta) != required and set(meta) != allowed:
        _fail(
            "B2_CACHE_RECORD_HASH_SCHEMA_INVALID",
            f"tensor {name} fields are not exact",
        )
    logical_name = meta["logical_name"]
    dtype_name = meta["dtype"]
    shape = meta["shape"]
    semantics = meta["dimension_semantics"]
    digest = meta["digest"]
    if logical_name != name:
        _fail(
            "B2_CACHE_RECORD_HASH_SCHEMA_INVALID",
            f"tensor key {name!r} does not match logical_name",
        )
    if dtype_name != "float32":
        _fail(
            "B2_CACHE_TENSOR_DTYPE_INVALID",
            f"primary scientific tensor {name} must be float32",
        )
    if (
        not isinstance(shape, list | tuple)
        or any(
            not isinstance(size, int) or isinstance(size, bool) or size < 1
            for size in shape
        )
        or not isinstance(semantics, list | tuple)
        or len(semantics) != len(shape)
        or any(not isinstance(item, str) or not item for item in semantics)
        or not _is_sha256(digest)
    ):
        _fail(
            "B2_CACHE_RECORD_HASH_SCHEMA_INVALID",
            f"tensor {name} metadata is invalid",
        )
    if "tensor" in meta:
        tensor = meta["tensor"]
        _validate_float32_tensor(
            tensor,
            expected_shape=tuple(int(size) for size in shape),
            role=f"persisted tensor {name}",
        )
        computed = canonical_tensor_digest(
            name,
            tensor,
            tuple(str(item) for item in semantics),
        )
        if computed != digest:
            _fail(
                "B2_CACHE_TENSOR_DIGEST_MISMATCH",
                f"persisted tensor {name} does not match its digest",
            )
    return {
        "logical_name": name,
        "dtype": dtype_name,
        "shape": [int(size) for size in shape],
        "dimension_semantics": [str(item) for item in semantics],
        "digest": digest,
    }


def _canonicalize_lattice(value: Any) -> list[dict[str, int]]:
    if not isinstance(value, list | tuple):
        _fail("B2_CACHE_RECORD_HASH_SCHEMA_INVALID", "causal map lattice invalid")
    rows: list[dict[str, int]] = []
    for item in value:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"checkpoint_depth", "candidate_layer_id"}
            or not isinstance(item["checkpoint_depth"], int)
            or isinstance(item["checkpoint_depth"], bool)
            or not isinstance(item["candidate_layer_id"], int)
            or isinstance(item["candidate_layer_id"], bool)
        ):
            _fail("B2_CACHE_RECORD_HASH_SCHEMA_INVALID", "lattice identity invalid")
        rows.append(
            {
                "checkpoint_depth": item["checkpoint_depth"],
                "candidate_layer_id": item["candidate_layer_id"],
            }
        )
    return sorted(
        rows,
        key=lambda row: (row["checkpoint_depth"], row["candidate_layer_id"]),
    )


def scientific_record_content(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract the explicit scientific whitelist for record hashing."""

    if not isinstance(record, Mapping):
        _fail("B2_CACHE_RECORD_HASH_SCHEMA_INVALID", "record must be an object")
    schema_version = record.get("record_hash_schema_version")
    if schema_version != _RECORD_HASH_SCHEMA_VERSION:
        _fail(
            "B2_CACHE_RECORD_HASH_SCHEMA_INVALID",
            "unsupported record hash schema version",
        )
    unknown = set(record) - set(_SCIENTIFIC_RECORD_FIELDS_V1) - _KNOWN_EXCLUDED_RECORD_FIELDS
    if unknown:
        _fail(
            "B2_CACHE_RECORD_HASH_SCHEMA_INVALID",
            f"unknown record fields: {sorted(unknown)}",
        )
    missing = [key for key in _SCIENTIFIC_RECORD_FIELDS_V1 if key not in record]
    if missing:
        _fail(
            "B2_CACHE_RECORD_HASH_SCHEMA_INVALID",
            f"missing scientific fields: {missing}",
        )

    tensors_raw = record["tensors"]
    if not isinstance(tensors_raw, Mapping) or not tensors_raw:
        _fail("B2_CACHE_RECORD_HASH_SCHEMA_INVALID", "tensors mapping is required")
    tensors = {
        name: _canonicalize_tensor_meta(name, tensors_raw[name])
        for name in sorted(tensors_raw)
    }
    feature_names = record["descriptor_feature_names"]
    if not isinstance(feature_names, list | tuple):
        _fail(
            "B2_CACHE_RECORD_HASH_SCHEMA_INVALID",
            "descriptor feature names invalid",
        )
    content: dict[str, Any] = {
        "record_schema_version": record["record_schema_version"],
        "record_hash_schema_version": record["record_hash_schema_version"],
        "stable_sample_id": record["stable_sample_id"],
        "membership": record["membership"],
        "category": record["category"],
        "image_label": record["image_label"],
        "anomaly_type": record["anomaly_type"],
        "image_identity": record["image_identity"],
        "mask_identity": record["mask_identity"],
        "candidate_layers": list(record["candidate_layers"]),
        "prediction_depths": list(record["prediction_depths"]),
        "causal_map_lattice": _canonicalize_lattice(record["causal_map_lattice"]),
        "cache_tensor_contract_version": record["cache_tensor_contract_version"],
        "tensors": tensors,
        "descriptor_contract_version": record["descriptor_contract_version"],
        "descriptor_feature_names": list(feature_names),
        "descriptor_source_tensor_kind": record["descriptor_source_tensor_kind"],
        "descriptor_implementation_sha256": record[
            "descriptor_implementation_sha256"
        ],
        "extractor_configuration_sha256": record[
            "extractor_configuration_sha256"
        ],
        "split_scientific_hash_version": record["split_scientific_hash_version"],
        "split_scientific_sha256": record["split_scientific_sha256"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "execution_profile_name": record["execution_profile_name"],
        "execution_profile_sha256": record["execution_profile_sha256"],
    }
    return MappingProxyType(content)


def record_scientific_sha256(record: Mapping[str, Any]) -> str:
    """SHA-256 over the explicit scientific whitelist content only."""

    return _canonical_sha256(dict(scientific_record_content(record)))


@dataclass(frozen=True)
class PersistedSampleEntry:
    """Verified Option A sample artifact identity for partial/final manifests."""

    stable_sample_id: str
    relative_path: str
    record_scientific_sha256: str
    record_file_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_relative_path(stable_sample_id: str) -> str:
    """Deterministic run-relative artifact path for a planned stable sample ID."""

    if not _is_sha256(stable_sample_id):
        _fail("B2_CACHE_SAMPLE_SCHEMA_INVALID", "stable sample ID is invalid")
    return f"samples/{stable_sample_id}.pt"


def claim_new_run_directory(run_dir: Path) -> None:
    """Refuse silent overwrite of an existing official run directory."""

    from rad.artifacts import refuse_existing_run
    from rad.errors import OutputProtectionError

    try:
        refuse_existing_run(run_dir)
    except OutputProtectionError as exc:
        _fail("B2_CACHE_RUN_EXISTS", str(exc))


def _persistable_scientific_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the hashed whitelist while retaining optional tensor bytes for Option A."""

    content = _thaw_for_json(scientific_record_content(record))
    source_tensors = _mapping(record["tensors"], "tensors")
    persisted_tensors: dict[str, Any] = {}
    for name, meta in content["tensors"].items():
        persisted = dict(meta)
        source = source_tensors[name]
        if isinstance(source, Mapping) and "tensor" in source:
            persisted["tensor"] = source["tensor"]
        persisted_tensors[name] = persisted
    content["tensors"] = persisted_tensors
    return content


def _verify_persisted_tensor_values(record: Mapping[str, Any]) -> None:
    """Verify every persisted tensor against its digest after a `.pt` reload."""

    tensors = _mapping(record.get("tensors"), "tensors")
    for name, meta in tensors.items():
        if not isinstance(meta, Mapping) or "tensor" not in meta:
            continue
        _canonicalize_tensor_meta(str(name), meta)


def write_sample_atomic(
    path: Path,
    scientific_record: Mapping[str, Any],
) -> PersistedSampleEntry:
    """Atomically persist an Option A `.pt` and return verified dual-hash entry."""

    torch = _bind_production_tensor_apis()
    destination = Path(path)
    content = dict(scientific_record_content(scientific_record))
    persisted_record = _persistable_scientific_record(scientific_record)
    scientific_digest = _canonical_sha256(content)
    stable_id = content["stable_sample_id"]
    if not _is_sha256(stable_id):
        _fail("B2_CACHE_SAMPLE_SCHEMA_INVALID", "stable sample ID is invalid")
    relative = sample_relative_path(stable_id)
    if destination.name != f"{stable_id}.pt":
        _fail(
            "B2_CACHE_SAMPLE_IDENTITY_INVALID",
            "destination filename must equal the stable sample ID",
        )
    if destination.exists():
        _fail(
            "B2_CACHE_SAMPLE_EXISTS",
            f"sample artifact already exists: {destination}",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scientific_record": persisted_record,
        "record_scientific_sha256": scientific_digest,
    }
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp_path, destination)
        except FileExistsError:
            _fail(
                "B2_CACHE_SAMPLE_EXISTS",
                f"sample artifact already exists: {destination}",
            )
        tmp_path.unlink(missing_ok=True)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    file_digest = _sha256_file(destination)
    loaded = torch.load(destination, map_location="cpu", weights_only=True)
    if not isinstance(loaded, Mapping) or set(loaded) != {
        "scientific_record",
        "record_scientific_sha256",
    }:
        _fail("B2_CACHE_PT_PAYLOAD_INVALID", "Option A payload keys are not exact")
    if loaded["record_scientific_sha256"] != scientific_digest:
        _fail(
            "B2_CACHE_RESUME_SCIENTIFIC_HASH_MISMATCH",
            "embedded scientific hash drifted after write",
        )
    recomputed = record_scientific_sha256(loaded["scientific_record"])
    if recomputed != scientific_digest:
        _fail(
            "B2_CACHE_RESUME_SCIENTIFIC_HASH_MISMATCH",
            "reloaded scientific record failed rehash",
        )
    _verify_persisted_tensor_values(loaded["scientific_record"])
    return PersistedSampleEntry(
        stable_sample_id=stable_id,
        relative_path=relative,
        record_scientific_sha256=scientific_digest,
        record_file_sha256=file_digest,
    )


def write_partial_manifest_atomic(
    path: Path,
    manifest: Mapping[str, Any],
) -> None:
    """Publish a partial manifest; never allow status=passed."""

    if not isinstance(manifest, Mapping) or not manifest:
        _fail("B2_CACHE_RESUME_MANIFEST_INVALID", "partial manifest is missing")
    if manifest.get("status") == "passed":
        _fail(
            "B2_CACHE_PARTIAL_STATUS_INVALID",
            "partial manifest cannot claim status=passed",
        )
    if manifest.get("status") != "partial":
        _fail(
            "B2_CACHE_PARTIAL_STATUS_INVALID",
            "partial manifest status must be 'partial'",
        )
    from rad.artifacts import atomic_write_json

    atomic_write_json(path, _thaw_for_json(manifest))


def _planned_sample_filenames(plan: Sequence[PlannedSample]) -> set[str]:
    planned_ids = [row.stable_sample_id for row in plan]
    planned_id_set = set(planned_ids)
    if len(planned_ids) != len(planned_id_set):
        _fail("B2_CACHE_COVERAGE_MISMATCH", "plan contains duplicate sample IDs")
    return {f"{sample_id}.pt" for sample_id in planned_id_set}


def audit_samples_directory_filenames(
    run_dir: Path,
    plan: Sequence[PlannedSample],
    *,
    require_complete: bool,
) -> None:
    """Audit samples/ against the plan-derived deterministic filename set."""

    expected_names = _planned_sample_filenames(plan)
    samples_dir = Path(run_dir) / "samples"
    if not samples_dir.is_dir():
        if require_complete:
            _fail("B2_CACHE_ORPHAN_ARTIFACT", "samples directory is missing")
        return
    actual_names = {path.name for path in samples_dir.iterdir()}
    extras = actual_names - expected_names
    if extras:
        _fail(
            "B2_CACHE_ORPHAN_ARTIFACT",
            f"unexpected samples-directory entries: {sorted(extras)}",
        )
    if require_complete:
        missing = expected_names - actual_names
        if missing:
            _fail(
                "B2_CACHE_COVERAGE_MISMATCH",
                f"missing planned sample artifacts: {sorted(missing)}",
            )


def _compare_run_provenance(
    partial_manifest: Mapping[str, Any],
    expected_run_provenance: Mapping[str, Any],
) -> None:
    if not isinstance(expected_run_provenance, Mapping) or not expected_run_provenance:
        _fail(
            "B2_CACHE_RESUME_PROVENANCE_DRIFT",
            "expected run provenance is required for resume",
        )
    required = (
        "split_scientific_sha256",
        "checkpoint_sha256",
        "execution_profile_name",
        "execution_profile_sha256",
        "runtime_attestation_sha256",
    )
    if any(key not in expected_run_provenance for key in required):
        _fail(
            "B2_CACHE_RESUME_PROVENANCE_DRIFT",
            "expected run provenance fields are incomplete",
        )
    for field in required:
        if partial_manifest.get(field) != expected_run_provenance[field]:
            _fail(
                "B2_CACHE_RESUME_PROVENANCE_DRIFT",
                f"run provenance field {field} drifted",
            )
    if "generation_commit" in partial_manifest:
        expected_commit = expected_run_provenance.get("generation_commit")
        if (
            not isinstance(expected_commit, str)
            or not _is_git_commit(expected_commit)
            or partial_manifest["generation_commit"] != expected_commit
        ):
            _fail(
                "B2_CACHE_RESUME_PROVENANCE_DRIFT",
                "generation commit drifted during resume",
            )


def validate_resume_state(
    run_dir: Path,
    partial_manifest: Mapping[str, Any],
    *,
    plan: Sequence[PlannedSample],
    expected_descriptor_contract: Mapping[str, Any],
    expected_run_provenance: Mapping[str, Any],
) -> tuple[PersistedSampleEntry, ...]:
    """Recompute hashes and provenance before reusing immutable samples."""

    torch = _bind_production_tensor_apis()
    if not isinstance(partial_manifest, Mapping) or not partial_manifest:
        _fail("B2_CACHE_RESUME_MANIFEST_INVALID", "partial manifest is missing")
    if partial_manifest.get("status") != "partial":
        _fail(
            "B2_CACHE_RESUME_MANIFEST_INVALID",
            "resume requires a partial manifest",
        )
    planned_ids = [row.stable_sample_id for row in plan]
    claimed_ids = partial_manifest.get("planned_stable_sample_ids")
    if claimed_ids != planned_ids:
        _fail(
            "B2_CACHE_RESUME_PROVENANCE_DRIFT",
            "partial plan identity drifted from the requested plan",
        )
    required_outer = (
        "split_scientific_sha256",
        "checkpoint_sha256",
        "execution_profile_name",
        "execution_profile_sha256",
        "runtime_attestation_sha256",
        "descriptor_contract",
        "samples",
    )
    if any(key not in partial_manifest for key in required_outer):
        _fail("B2_CACHE_RESUME_MANIFEST_INVALID", "partial manifest fields incomplete")
    for field in (
        "split_scientific_sha256",
        "checkpoint_sha256",
        "execution_profile_sha256",
        "runtime_attestation_sha256",
    ):
        if not _is_sha256(partial_manifest[field]):
            _fail(
                "B2_CACHE_RESUME_MANIFEST_INVALID",
                f"partial manifest field {field} is invalid",
            )
    if (
        not isinstance(partial_manifest["execution_profile_name"], str)
        or not partial_manifest["execution_profile_name"]
    ):
        _fail(
            "B2_CACHE_RESUME_MANIFEST_INVALID",
            "execution profile name is invalid",
        )
    _compare_run_provenance(partial_manifest, expected_run_provenance)
    try:
        validate_descriptor_contract(
            partial_manifest["descriptor_contract"],
            expected_descriptor_contract,
        )
    except TeacherCacheError as exc:
        if exc.code == "B2_CACHE_DESCRIPTOR_CONTRACT_DRIFT":
            _fail(
                "B2_CACHE_RESUME_PROVENANCE_DRIFT",
                "descriptor contract drifted during resume",
            )
        raise

    audit_samples_directory_filenames(run_dir, plan, require_complete=False)

    samples = partial_manifest["samples"]
    if not isinstance(samples, list):
        _fail("B2_CACHE_RESUME_MANIFEST_INVALID", "samples must be a list")
    plan_id_set = set(planned_ids)
    entries: list[PersistedSampleEntry] = []
    seen: set[str] = set()
    root = Path(run_dir)
    for raw in samples:
        if not isinstance(raw, Mapping):
            _fail("B2_CACHE_RESUME_MANIFEST_INVALID", "sample entry must be an object")
        required = {
            "stable_sample_id",
            "relative_path",
            "record_scientific_sha256",
            "record_file_sha256",
        }
        if set(raw) != required:
            _fail("B2_CACHE_RESUME_MANIFEST_INVALID", "sample entry fields are not exact")
        stable_id = raw["stable_sample_id"]
        relative = raw["relative_path"]
        claimed_scientific = raw["record_scientific_sha256"]
        claimed_file = raw["record_file_sha256"]
        if (
            not _is_sha256(stable_id)
            or stable_id not in plan_id_set
            or stable_id in seen
            or relative != sample_relative_path(stable_id)
            or not _is_sha256(claimed_scientific)
            or not _is_sha256(claimed_file)
        ):
            _fail(
                "B2_CACHE_RESUME_PROVENANCE_DRIFT",
                f"sample entry {stable_id!r} is not reusable",
            )
        seen.add(stable_id)
        artifact = root / relative
        if not artifact.is_file():
            _fail(
                "B2_CACHE_RESUME_FILE_HASH_MISMATCH",
                f"sample artifact missing: {artifact}",
            )
        file_digest = _sha256_file(artifact)
        if file_digest != claimed_file:
            _fail(
                "B2_CACHE_RESUME_FILE_HASH_MISMATCH",
                f"sample file hash drifted for {stable_id}",
            )
        loaded = torch.load(artifact, map_location="cpu", weights_only=True)
        if not isinstance(loaded, Mapping) or set(loaded) != {
            "scientific_record",
            "record_scientific_sha256",
        }:
            _fail("B2_CACHE_PT_PAYLOAD_INVALID", "Option A payload keys are not exact")
        embedded = loaded["record_scientific_sha256"]
        recomputed = record_scientific_sha256(loaded["scientific_record"])
        if (
            recomputed != embedded
            or recomputed != claimed_scientific
            or embedded != claimed_scientific
        ):
            _fail(
                "B2_CACHE_RESUME_SCIENTIFIC_HASH_MISMATCH",
                f"scientific hash drifted for {stable_id}",
            )
        if loaded["scientific_record"].get("stable_sample_id") != stable_id:
            _fail(
                "B2_CACHE_RESUME_PROVENANCE_DRIFT",
                f"embedded sample identity drifted for {stable_id}",
            )
        entries.append(
            PersistedSampleEntry(
                stable_sample_id=stable_id,
                relative_path=relative,
                record_scientific_sha256=recomputed,
                record_file_sha256=file_digest,
            )
        )
    return tuple(entries)


def audit_complete_coverage(
    run_dir: Path,
    plan: Sequence[PlannedSample],
    entries: Sequence[PersistedSampleEntry],
) -> None:
    """Require exact plan/entry/file identity sets under samples/."""

    planned_ids = [row.stable_sample_id for row in plan]
    planned_id_set = set(planned_ids)
    if len(planned_ids) != len(planned_id_set):
        _fail("B2_CACHE_COVERAGE_MISMATCH", "plan contains duplicate sample IDs")
    entry_ids = [entry.stable_sample_id for entry in entries]
    if len(entry_ids) != len(set(entry_ids)) or set(entry_ids) != planned_id_set:
        _fail("B2_CACHE_COVERAGE_MISMATCH", "entries do not exactly match the plan")
    for entry in entries:
        if entry.relative_path != sample_relative_path(entry.stable_sample_id):
            _fail(
                "B2_CACHE_COVERAGE_MISMATCH",
                f"entry path mapping drifted for {entry.stable_sample_id}",
            )
    audit_samples_directory_filenames(run_dir, plan, require_complete=True)


def build_final_manifest(
    *,
    partial_manifest: Mapping[str, Any],
    entries: Sequence[PersistedSampleEntry],
    plan: Sequence[PlannedSample],
) -> Mapping[str, Any]:
    """Build a passed final manifest from verified immutable sample entries."""

    if not isinstance(partial_manifest, Mapping) or not partial_manifest:
        _fail("B2_CACHE_RESUME_MANIFEST_INVALID", "partial manifest is missing")
    if partial_manifest.get("status") == "passed":
        _fail(
            "B2_CACHE_PARTIAL_STATUS_INVALID",
            "finalization input must not already be passed",
        )
    planned_ids = [row.stable_sample_id for row in plan]
    entry_ids = [entry.stable_sample_id for entry in entries]
    if entry_ids != planned_ids:
        _fail(
            "B2_CACHE_COVERAGE_MISMATCH",
            "final manifest entries must match plan order exactly",
        )
    for entry in entries:
        if entry.relative_path != sample_relative_path(entry.stable_sample_id):
            _fail(
                "B2_CACHE_COVERAGE_MISMATCH",
                f"final entry path mapping drifted for {entry.stable_sample_id}",
            )
        if not _is_sha256(entry.record_scientific_sha256) or not _is_sha256(
            entry.record_file_sha256
        ):
            _fail("B2_CACHE_COVERAGE_MISMATCH", "final entry hashes are invalid")
    payload = _thaw_for_json(partial_manifest)
    payload["status"] = "passed"
    payload["planned_stable_sample_ids"] = list(planned_ids)
    payload["samples"] = [
        {
            "stable_sample_id": entry.stable_sample_id,
            "relative_path": entry.relative_path,
            "record_scientific_sha256": entry.record_scientific_sha256,
            "record_file_sha256": entry.record_file_sha256,
        }
        for entry in entries
    ]
    payload["sample_coverage_sha256"] = _canonical_sha256(
        {"planned_stable_sample_ids": list(planned_ids)}
    )
    payload["cache_scientific_sha256"] = _canonical_sha256(
        {
            "planned_stable_sample_ids": list(planned_ids),
            "record_scientific_sha256_by_id": {
                entry.stable_sample_id: entry.record_scientific_sha256
                for entry in entries
            },
        }
    )
    return MappingProxyType(payload)

def _tensor_meta_with_value(name: str, tensor: Any) -> dict[str, Any]:
    value = tensor.detach().to(device="cpu", dtype=_bind_production_tensor_apis().float32).contiguous()
    semantics = (
        ("scalar",)
        if value.ndim == 1
        else ("batch", "channel", "height", "width")
    )
    return {
        "logical_name": name,
        "dtype": "float32",
        "shape": list(value.shape),
        "dimension_semantics": list(semantics),
        "digest": canonical_tensor_digest(name, value, semantics),
        "tensor": value,
    }


def build_scientific_record(
    *,
    sample: PlannedSample,
    validated: ValidatedTeacherOutput,
    cumulative: Mapping[int, Any],
    image_score: Any,
    config: TeacherCacheConfig,
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    for identity in sorted(validated.maps):
        name = f"causal_map:{identity.checkpoint_depth}:{identity.candidate_layer_id}"
        tensors[name] = _tensor_meta_with_value(name, validated.maps[identity])
    for depth, tensor in cumulative.items():
        name = f"cumulative_map:{depth}"
        tensors[name] = _tensor_meta_with_value(name, tensor)
    tensors["full_depth_map"] = _tensor_meta_with_value(
        "full_depth_map", cumulative[max(cumulative)]
    )
    if validated.anomalous_mask is not None:
        tensors["anomalous_mask"] = _tensor_meta_with_value(
            "anomalous_mask", validated.anomalous_mask
        )
    tensors["image_score"] = _tensor_meta_with_value("image_score", image_score)
    return {
        "record_schema_version": 1,
        "record_hash_schema_version": config.record_hash_schema_version,
        "stable_sample_id": sample.stable_sample_id,
        "membership": sample.membership,
        "category": sample.category,
        "image_label": sample.image_label,
        "anomaly_type": sample.anomaly_type,
        "image_identity": sample.image_identity,
        "mask_identity": sample.mask_identity,
        "candidate_layers": list(config.candidate_layers),
        "prediction_depths": list(config.prediction_depths),
        "causal_map_lattice": [
            {
                "checkpoint_depth": item.checkpoint_depth,
                "candidate_layer_id": item.candidate_layer_id,
            }
            for item in sorted(validated.maps)
        ],
        "cache_tensor_contract_version": config.cache_tensor_contract_version,
        "tensors": tensors,
        "descriptor_contract_version": descriptor["descriptor_contract_version"],
        "descriptor_feature_names": list(descriptor["feature_names"]),
        "descriptor_source_tensor_kind": descriptor["descriptor_source_tensor_kind"],
        "descriptor_implementation_sha256": descriptor["descriptor_implementation_sha256"],
        "extractor_configuration_sha256": descriptor["extractor_configuration_sha256"],
        "split_scientific_hash_version": config.split_scientific_hash_version,
        "split_scientific_sha256": config.split_scientific_sha256,
        "checkpoint_sha256": config.checkpoint_sha256,
        "execution_profile_name": config.execution_profile_name,
        "execution_profile_sha256": config.execution_profile_sha256,
    }

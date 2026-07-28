"""B2-03A descriptor artifact schema and source-only normalization contracts.

Pure logic only: no VisualAD model load, no Git inspection, no output-directory
selection, no global runtime mutation, no VisA access, and no silent writes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, NoReturn

import torch

import rad.phase_b.b2_teacher_cache as cache_mod
from rad.models.descriptors import LAYER_DESCRIPTOR_FEATURE_NAMES

_REPO_ROOT = Path(__file__).resolve().parents[2]

_EXPECTED_MAIN_TAG = "b2-main-integration-v1"
_EXPECTED_MAIN_COMMIT = "51e18ade0231c7488ef582bde1e9694f933e85eb"
_EXPECTED_SPLIT_SHA256 = (
    "91570da1fed6d7859d407196b10403581832ae0ff677a1ea7657ca76b91471f0"
)
_EXPECTED_CHECKPOINT_SHA256 = (
    "97bd461163efb96e36cddb1c3adf677e4c4fc2daabb2521021689f30e799b4f4"
)
_EXPECTED_PROFILE_SHA256 = (
    "7af8dba39633743da0380fef9710940cded655f68c9efa8f84f5a52aeddb3c8d"
)
_EXPECTED_CANDIDATE_LAYERS = (6, 12, 18, 24)
_EXPECTED_PREDICTION_DEPTHS = (12, 18, 24)
_EXPECTED_DESCRIPTOR_DIMENSION = 18
_EXPECTED_DESCRIPTOR_CONTRACT_VERSION = 1
_EXPECTED_NORMALIZATION_CONTRACT_VERSION = 1
_EXPECTED_SPLIT_COUNTS = {"training": 16, "calibration": 8, "evaluation": 8}
_EXPECTED_PRIMARY_DTYPE = "float32"
_EXPECTED_CONFIGURATION_ID = "b2_descriptor_artifacts_gate_c"
_EXPECTED_SCHEMA_VERSION = 1
_DESCRIPTOR_SEMANTICS = ("batch", "layer", "feature")
_FEATURE_ORDER = tuple(LAYER_DESCRIPTOR_FEATURE_NAMES)

_DESCRIPTOR_RECORD_SCIENTIFIC_FIELDS: tuple[str, ...] = (
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
    "teacher_cache_scientific_sha256",
    "split_scientific_sha256",
    "checkpoint_sha256",
    "execution_profile_sha256",
)

_KNOWN_EXCLUDED_DESCRIPTOR_FIELDS = frozenset(
    {
        "descriptor_record_scientific_sha256",
        "source_teacher_record_file_sha256",
        "record_file_sha256",
        "absolute_output_path",
        "git_branch",
        "worktree_path",
        "timestamp",
        "runtime_attestation_sha256",
        "output_path",
        "generation_commit",
        "generation_branch",
        "worktree_clean",
        "machine_hostname",
        "environment",
        "run_id",
    }
)

_NORMALIZATION_SCIENTIFIC_FIELDS: tuple[str, ...] = (
    "normalization_contract_version",
    "ordered_training_stable_sample_ids",
    "descriptor_record_scientific_sha256_by_id",
    "axes",
    "training_sample_coverage_sha256",
    "normalization_training_coverage_sha256",
    "statistics_dtype",
    "standard_deviation_ddof",
)

_FINAL_MANIFEST_NAME = "final_manifest.json"
_FINAL_MANIFEST_RECEIPT_NAME = "final_manifest.json.sha256"
_NORMALIZATION_RELATIVE_PATH = "normalization_statistics.pt"


class DescriptorArtifactsError(RuntimeError):
    """A descriptor-artifacts contract failure carrying a stable error code."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise DescriptorArtifactsError(code, detail)


@dataclass(frozen=True)
class DescriptorArtifactsConfig:
    schema_version: int
    configuration_id: str
    expected_main_tag: str
    expected_main_commit: str
    expected_teacher_cache_scientific_sha256: str
    expected_sample_coverage_sha256: str
    expected_split_scientific_sha256: str
    expected_checkpoint_sha256: str
    expected_execution_profile_sha256: str
    candidate_layers: tuple[int, ...]
    prediction_depths: tuple[int, ...]
    descriptor_dimension: int
    descriptor_contract_version: int
    normalization_contract_version: int
    required_split_counts: Mapping[str, int]
    primary_dtype: str
    fail_closed_requirements: tuple[str, ...]


@dataclass(frozen=True)
class PlannedDescriptorSample:
    stable_sample_id: str
    membership: str


@dataclass(frozen=True)
class AcceptedSampleEntry:
    stable_sample_id: str
    relative_path: str
    record_scientific_sha256: str
    record_file_sha256: str
    membership: str


@dataclass(frozen=True)
class AcceptedTeacherCache:
    manifest: Mapping[str, Any]
    plan: tuple[PlannedDescriptorSample, ...]
    entries: tuple[AcceptedSampleEntry, ...]


@dataclass(frozen=True)
class DiskAuthoritativeTeacherCacheManifest:
    manifest: Mapping[str, Any]
    manifest_path: Path
    cache_root: Path
    source_teacher_cache_manifest_file_sha256: str


@dataclass(frozen=True)
class ValidatedTeacherCache:
    manifest: Mapping[str, Any]
    manifest_path: Path
    cache_root: Path
    source_teacher_cache_manifest_file_sha256: str
    accepted: AcceptedTeacherCache


@dataclass(frozen=True)
class PersistedDescriptorEntry:
    stable_sample_id: str
    relative_record_path: str
    descriptor_record_scientific_sha256: str
    descriptor_record_file_sha256: str
    verification_status: str = "verified"


@dataclass(frozen=True)
class PersistedNormalizationEntry:
    relative_path: str
    normalization_statistics_scientific_sha256: str
    normalization_statistics_file_sha256: str


@dataclass(frozen=True)
class DescriptorCollectionResult:
    run_dir: Path
    manifest: Mapping[str, Any]
    source_teacher_cache_manifest_file_sha256: str
    teacher_forward_count: int


@dataclass(frozen=True)
class VerifiedDescriptorCollection:
    run_dir: Path
    manifest: Mapping[str, Any]
    descriptor_records_by_id: Mapping[str, Mapping[str, Any]]
    normalization_statistics: Mapping[str, Any]
    teacher_forward_count: int


@dataclass(frozen=True)
class DescriptorCollectionComparison:
    scientifically_equivalent: bool
    reasons: tuple[str, ...]
    file_byte_equal: bool


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        ch in "0123456789abcdef" for ch in value
    )


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_within_root(root: Path, relative_path: str, *, code: str) -> Path:
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or Path(relative_path).is_absolute()
    ):
        _fail(code, f"path must be a non-empty relative path: {relative_path!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative_path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        _fail(code, f"path escapes cache root: {relative_path}")
    return resolved


def _normalized_relative_path(relative_path: str, *, code: str) -> str:
    if not isinstance(relative_path, str) or not relative_path:
        _fail(code, "path must be a non-empty relative path")
    raw = PurePosixPath(relative_path)
    if raw.is_absolute():
        _fail(code, f"path must be relative: {relative_path}")
    parts = raw.parts
    if not parts or any(part in {"", ".."} for part in parts):
        _fail(code, f"path escapes root or is malformed: {relative_path}")
    normalized_parts = tuple(part for part in parts if part != ".")
    if not normalized_parts:
        _fail(code, f"path must not normalize to the run root: {relative_path}")
    return PurePosixPath(*normalized_parts).as_posix()


def resolve_run_relative_artifact(
    *,
    run_dir: Path,
    relative_path: str,
    expected_kind: str,
) -> Path:
    normalized = _normalized_relative_path(
        relative_path,
        code="B2_DESC_RUN_RELATIVE_PATH_INVALID",
    )
    resolved_root = Path(run_dir).resolve()
    candidate = resolved_root / normalized
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        _fail(
            "B2_DESC_MISSING_ARTIFACT",
            f"{expected_kind} missing or unreadable at {normalized}: {exc}",
        )
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        _fail(
            "B2_DESC_RUN_ROOT_ESCAPE",
            f"{expected_kind} escapes authoritative run_dir: {relative_path}",
        )
    if not resolved.is_file():
        _fail(
            "B2_DESC_MISSING_ARTIFACT",
            f"{expected_kind} must be a regular file inside run_dir: {normalized}",
        )
    return resolved


def _run_relative_path_from_resolved(run_dir: Path, path: Path) -> str:
    return path.resolve().relative_to(Path(run_dir).resolve()).as_posix()


def load_disk_authoritative_teacher_cache_manifest(
    *,
    teacher_cache_manifest_path: Path | str,
    teacher_cache_root: Path | str,
) -> DiskAuthoritativeTeacherCacheManifest:
    """Load the declared teacher-cache manifest from disk after root binding."""

    manifest_path = Path(teacher_cache_manifest_path)
    cache_root = Path(teacher_cache_root)
    resolved_root = cache_root.resolve()
    resolved_manifest = manifest_path.resolve()
    try:
        resolved_manifest.relative_to(resolved_root)
    except ValueError:
        _fail(
            "B2_DESC_CACHE_MANIFEST_OUTSIDE_ROOT",
            "teacher-cache manifest must resolve inside teacher-cache root",
        )
    try:
        payload = manifest_path.read_bytes()
    except OSError as exc:
        _fail("B2_DESC_CACHE_MANIFEST_INVALID", f"cannot read manifest bytes: {exc}")
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        _fail("B2_DESC_CACHE_MANIFEST_INVALID", f"invalid manifest JSON: {exc}")
    if not isinstance(manifest, Mapping):
        _fail("B2_DESC_CACHE_MANIFEST_INVALID", "teacher-cache manifest must be an object")
    return DiskAuthoritativeTeacherCacheManifest(
        manifest=MappingProxyType(dict(manifest)),
        manifest_path=resolved_manifest,
        cache_root=resolved_root,
        source_teacher_cache_manifest_file_sha256=hashlib.sha256(payload).hexdigest(),
    )


def load_and_validate_accepted_teacher_cache_from_disk(
    *,
    config: DescriptorArtifactsConfig,
    teacher_cache_manifest_path: Path,
    teacher_cache_root: Path,
) -> ValidatedTeacherCache:
    authoritative = load_disk_authoritative_teacher_cache_manifest(
        teacher_cache_manifest_path=teacher_cache_manifest_path,
        teacher_cache_root=teacher_cache_root,
    )
    accepted = validate_accepted_teacher_cache(
        manifest=authoritative.manifest,
        config=config,
        cache_root=authoritative.cache_root,
        allow_test_fixture=False,
    )
    return ValidatedTeacherCache(
        manifest=authoritative.manifest,
        manifest_path=authoritative.manifest_path,
        cache_root=authoritative.cache_root,
        source_teacher_cache_manifest_file_sha256=(
            authoritative.source_teacher_cache_manifest_file_sha256
        ),
        accepted=accepted,
    )


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("B2_DESC_CONFIG_INVALID", f"{field} must be an int")
    return value


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("B2_DESC_CONFIG_INVALID", f"{field} must be a non-empty string")
    return value


def _int_tuple(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, list | tuple) or not value:
        _fail("B2_DESC_CONFIG_INVALID", f"{field} must be a non-empty sequence")
    out: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            _fail("B2_DESC_CONFIG_INVALID", f"{field} entries must be ints")
        out.append(item)
    return tuple(out)


def _layers_for_depth(candidate_layers: Sequence[int], depth: int) -> tuple[int, ...]:
    return tuple(layer for layer in candidate_layers if layer <= depth)


def _expected_descriptor_shape(
    *,
    candidate_layers: Sequence[int],
    depth: int,
    descriptor_dimension: int,
) -> tuple[int, int, int]:
    return (1, len(_layers_for_depth(candidate_layers, depth)), descriptor_dimension)


def _authoritative_extractor_digests(
    *,
    descriptor_contract_version: int,
    repo_root: Path = _REPO_ROOT,
) -> tuple[str, str]:
    implementation = cache_mod.descriptor_implementation_sha256(repo_root)
    configuration = cache_mod._extractor_configuration_sha256(  # noqa: SLF001
        feature_names=_FEATURE_ORDER,
        descriptor_contract_version=descriptor_contract_version,
    )
    return implementation, configuration


def load_descriptor_artifacts_config(path: Path | str) -> DescriptorArtifactsConfig:
    """Load and pin the Gate-C descriptor-artifacts configuration."""

    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _fail("B2_DESC_CONFIG_MISSING", f"path does not exist: {config_path}")
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        _fail("B2_DESC_CONFIG_INVALID", f"cannot load config {config_path}: {exc}")
    if not isinstance(raw, dict):
        _fail("B2_DESC_CONFIG_INVALID", "config root must be an object")

    counts_raw = raw.get("required_split_counts")
    if not isinstance(counts_raw, Mapping):
        _fail("B2_DESC_CONFIG_INVALID", "required_split_counts must be an object")
    counts = {
        str(key): _require_int(value, f"required_split_counts.{key}")
        for key, value in counts_raw.items()
    }
    requirements_raw = raw.get("fail_closed_requirements")
    if not isinstance(requirements_raw, list | tuple):
        _fail("B2_DESC_CONFIG_INVALID", "fail_closed_requirements must be a list")
    requirements = tuple(str(item) for item in requirements_raw)

    config = DescriptorArtifactsConfig(
        schema_version=_require_int(raw.get("schema_version"), "schema_version"),
        configuration_id=_require_str(raw.get("configuration_id"), "configuration_id"),
        expected_main_tag=_require_str(raw.get("expected_main_tag"), "expected_main_tag"),
        expected_main_commit=_require_str(
            raw.get("expected_main_commit"), "expected_main_commit"
        ),
        expected_teacher_cache_scientific_sha256=_require_str(
            raw.get("expected_teacher_cache_scientific_sha256"),
            "expected_teacher_cache_scientific_sha256",
        ),
        expected_sample_coverage_sha256=_require_str(
            raw.get("expected_sample_coverage_sha256"),
            "expected_sample_coverage_sha256",
        ),
        expected_split_scientific_sha256=_require_str(
            raw.get("expected_split_scientific_sha256"),
            "expected_split_scientific_sha256",
        ),
        expected_checkpoint_sha256=_require_str(
            raw.get("expected_checkpoint_sha256"), "expected_checkpoint_sha256"
        ),
        expected_execution_profile_sha256=_require_str(
            raw.get("expected_execution_profile_sha256"),
            "expected_execution_profile_sha256",
        ),
        candidate_layers=_int_tuple(raw.get("candidate_layers"), "candidate_layers"),
        prediction_depths=_int_tuple(raw.get("prediction_depths"), "prediction_depths"),
        descriptor_dimension=_require_int(
            raw.get("descriptor_dimension"), "descriptor_dimension"
        ),
        descriptor_contract_version=_require_int(
            raw.get("descriptor_contract_version"), "descriptor_contract_version"
        ),
        normalization_contract_version=_require_int(
            raw.get("normalization_contract_version"),
            "normalization_contract_version",
        ),
        required_split_counts=MappingProxyType(counts),
        primary_dtype=_require_str(raw.get("primary_dtype"), "primary_dtype"),
        fail_closed_requirements=requirements,
    )
    _validate_pinned_config(config)
    return config


def _validate_pinned_config(config: DescriptorArtifactsConfig) -> None:
    if (
        config.schema_version != _EXPECTED_SCHEMA_VERSION
        or config.configuration_id != _EXPECTED_CONFIGURATION_ID
        or config.expected_main_tag != _EXPECTED_MAIN_TAG
        or config.expected_main_commit != _EXPECTED_MAIN_COMMIT
        or config.expected_split_scientific_sha256 != _EXPECTED_SPLIT_SHA256
        or config.expected_checkpoint_sha256 != _EXPECTED_CHECKPOINT_SHA256
        or config.expected_execution_profile_sha256 != _EXPECTED_PROFILE_SHA256
        or config.candidate_layers != _EXPECTED_CANDIDATE_LAYERS
        or config.prediction_depths != _EXPECTED_PREDICTION_DEPTHS
        or config.descriptor_dimension != _EXPECTED_DESCRIPTOR_DIMENSION
        or config.descriptor_contract_version != _EXPECTED_DESCRIPTOR_CONTRACT_VERSION
        or config.normalization_contract_version
        != _EXPECTED_NORMALIZATION_CONTRACT_VERSION
        or dict(config.required_split_counts) != _EXPECTED_SPLIT_COUNTS
        or config.primary_dtype != _EXPECTED_PRIMARY_DTYPE
        or "candidate_layers_configuration_driven"
        not in config.fail_closed_requirements
        or not _is_sha256(config.expected_teacher_cache_scientific_sha256)
        or not _is_sha256(config.expected_sample_coverage_sha256)
    ):
        _fail("B2_DESC_CONFIG_DRIFT", "descriptor-artifacts Gate-C config drifted")


def validate_accepted_teacher_cache(
    *,
    manifest: Mapping[str, Any],
    config: DescriptorArtifactsConfig,
    cache_root: Path | str,
    allow_test_fixture: bool = False,
) -> AcceptedTeacherCache:
    """Fail-closed acceptance of a passed teacher-cache for descriptor extraction."""

    root = Path(cache_root)
    if manifest.get("status") != "passed":
        _fail(
            "B2_DESC_CACHE_STATUS_NOT_PASSED",
            f"teacher-cache status must be passed, got {manifest.get('status')!r}",
        )
    claimed_scientific = manifest.get("cache_scientific_sha256")
    if claimed_scientific != config.expected_teacher_cache_scientific_sha256:
        _fail(
            "B2_DESC_CACHE_SCIENTIFIC_HASH_MISMATCH",
            "teacher-cache scientific hash does not match config expectation",
        )
    claimed_coverage = manifest.get("sample_coverage_sha256")
    if claimed_coverage != config.expected_sample_coverage_sha256:
        _fail(
            "B2_DESC_CACHE_COVERAGE_HASH_MISMATCH",
            "teacher-cache sample coverage hash does not match config expectation",
        )
    samples = manifest.get("samples")
    expected_count = sum(config.required_split_counts.values())
    if not isinstance(samples, list) or len(samples) != expected_count:
        _fail(
            "B2_DESC_CACHE_RECORD_COUNT_MISMATCH",
            f"expected {expected_count} records, got "
            f"{len(samples) if isinstance(samples, list) else type(samples)}",
        )
    if manifest.get("artifact_kind") == "test_fixture" and not allow_test_fixture:
        _fail(
            "B2_DESC_CACHE_TEST_FIXTURE_FORBIDDEN",
            "test_fixture teacher-cache is forbidden without allow_test_fixture",
        )
    if (
        manifest.get("split_scientific_sha256") is not None
        and manifest.get("split_scientific_sha256") != config.expected_split_scientific_sha256
    ):
        _fail(
            "B2_DESC_COLLECTION_PROVENANCE_MISMATCH",
            "teacher-cache manifest split identity does not match config",
        )
    if (
        manifest.get("checkpoint_sha256") is not None
        and manifest.get("checkpoint_sha256") != config.expected_checkpoint_sha256
    ):
        _fail(
            "B2_DESC_COLLECTION_PROVENANCE_MISMATCH",
            "teacher-cache manifest checkpoint identity does not match config",
        )
    if (
        manifest.get("execution_profile_sha256") is not None
        and manifest.get("execution_profile_sha256")
        != config.expected_execution_profile_sha256
    ):
        _fail(
            "B2_DESC_COLLECTION_PROVENANCE_MISMATCH",
            "teacher-cache manifest execution profile identity does not match config",
        )
    descriptor_contract = manifest.get("descriptor_contract")
    if isinstance(descriptor_contract, Mapping):
        if (
            descriptor_contract.get("descriptor_contract_version")
            != config.descriptor_contract_version
            or tuple(descriptor_contract.get("feature_names") or ()) != _FEATURE_ORDER
            or descriptor_contract.get("extractor_configuration_sha256")
            != _authoritative_extractor_digests(
                descriptor_contract_version=config.descriptor_contract_version
            )[1]
            or descriptor_contract.get("descriptor_implementation_sha256")
            != _authoritative_extractor_digests(
                descriptor_contract_version=config.descriptor_contract_version
            )[0]
        ):
            _fail(
                "B2_DESC_COLLECTION_PROVENANCE_MISMATCH",
                "teacher-cache manifest descriptor contract drifted",
            )

    expected_impl, expected_cfg = _authoritative_extractor_digests(
        descriptor_contract_version=config.descriptor_contract_version
    )
    plan: list[PlannedDescriptorSample] = []
    entries: list[AcceptedSampleEntry] = []
    membership_counts = {"training": 0, "calibration": 0, "evaluation": 0}
    seen_ids: set[str] = set()
    common_provenance: dict[str, Any] | None = None
    for raw_entry in samples:
        if not isinstance(raw_entry, Mapping):
            _fail("B2_DESC_CACHE_RECORD_HASH_MISMATCH", "sample entry must be an object")
        stable_id = str(raw_entry.get("stable_sample_id", ""))
        relative = str(raw_entry.get("relative_path") or cache_mod.sample_relative_path(stable_id))
        artifact = _resolve_within_root(
            root,
            relative,
            code="B2_DESC_CACHE_PATH_ESCAPE",
        )
        claimed_file = raw_entry.get("record_file_sha256")
        claimed_record = raw_entry.get("record_scientific_sha256")
        if not artifact.is_file():
            _fail(
                "B2_DESC_CACHE_FILE_HASH_MISMATCH",
                f"missing teacher-cache sample file: {relative}",
            )
        file_digest = _sha256_file(artifact)
        if file_digest != claimed_file:
            _fail(
                "B2_DESC_CACHE_FILE_HASH_MISMATCH",
                f"file hash mismatch for {stable_id}",
            )
        loaded = torch.load(artifact, map_location="cpu", weights_only=True)
        if not isinstance(loaded, Mapping):
            _fail("B2_DESC_CACHE_RECORD_HASH_MISMATCH", f"invalid payload for {stable_id}")
        scientific_record = loaded.get("scientific_record")
        embedded = loaded.get("record_scientific_sha256")
        if not isinstance(scientific_record, Mapping):
            _fail(
                "B2_DESC_CACHE_RECORD_HASH_MISMATCH",
                f"missing scientific_record for {stable_id}",
            )
        recomputed = cache_mod.record_scientific_sha256(scientific_record)
        if (
            recomputed != claimed_record
            or embedded != claimed_record
            or recomputed != embedded
        ):
            _fail(
                "B2_DESC_CACHE_RECORD_HASH_MISMATCH",
                f"record scientific hash mismatch for {stable_id}",
            )
        feature_names = scientific_record.get("descriptor_feature_names")
        if tuple(feature_names or ()) != _FEATURE_ORDER:
            _fail(
                "B2_DESC_FEATURE_ORDER_MISMATCH",
                f"descriptor feature order drifted for {stable_id}",
            )
        if (
            int(scientific_record.get("descriptor_contract_version", -1))
            != config.descriptor_contract_version
            or scientific_record.get("descriptor_implementation_sha256") != expected_impl
            or scientific_record.get("extractor_configuration_sha256") != expected_cfg
        ):
            _fail(
                "B2_DESC_CONTRACT_IDENTITY_MISMATCH",
                f"descriptor contract identity drifted for {stable_id}",
            )
        if (
            tuple(int(value) for value in scientific_record.get("candidate_layers", ()))
            != tuple(config.candidate_layers)
            or tuple(int(value) for value in scientific_record.get("prediction_depths", ()))
            != tuple(config.prediction_depths)
            or str(scientific_record.get("checkpoint_sha256"))
            != config.expected_checkpoint_sha256
            or str(scientific_record.get("execution_profile_sha256"))
            != config.expected_execution_profile_sha256
            or str(scientific_record.get("split_scientific_sha256"))
            != config.expected_split_scientific_sha256
        ):
            _fail(
                "B2_DESC_COLLECTION_PROVENANCE_MISMATCH",
                f"teacher-cache record provenance drifted for {stable_id}",
            )
        try:
            cache_mod.reconstruct_persisted_descriptors(scientific_record)
        except cache_mod.TeacherCacheError as exc:
            detail = str(exc)
            if "TENSOR_VALUE_MISSING" in detail or "MAP_" in detail:
                _fail(
                    "B2_DESC_DEPTH_MISSING",
                    f"required causal-map lattice incomplete for {stable_id}: {detail}",
                )
            _fail("B2_DESC_DEPTH_MISSING", f"descriptor reconstruction failed: {detail}")

        membership = str(scientific_record["membership"])
        if membership not in membership_counts:
            _fail(
                "B2_DESC_CACHE_SPLIT_COUNT_MISMATCH",
                f"unknown membership for {stable_id}: {membership!r}",
            )
        if stable_id in seen_ids:
            _fail("B2_DESC_CACHE_SPLIT_COUNT_MISMATCH", f"duplicate stable_sample_id {stable_id}")
        seen_ids.add(stable_id)
        membership_counts[membership] += 1
        provenance = {
            "checkpoint_sha256": str(scientific_record["checkpoint_sha256"]),
            "execution_profile_sha256": str(scientific_record["execution_profile_sha256"]),
            "split_scientific_sha256": str(scientific_record["split_scientific_sha256"]),
            "descriptor_contract_version": int(scientific_record["descriptor_contract_version"]),
            "descriptor_extractor_config_sha256": str(
                scientific_record["extractor_configuration_sha256"]
            ),
            "descriptor_extractor_implementation_sha256": str(
                scientific_record["descriptor_implementation_sha256"]
            ),
            "candidate_layers": tuple(int(value) for value in scientific_record["candidate_layers"]),
            "prediction_depths": tuple(int(value) for value in scientific_record["prediction_depths"]),
            "descriptor_feature_order": tuple(scientific_record["descriptor_feature_names"]),
        }
        if common_provenance is None:
            common_provenance = provenance
        elif provenance != common_provenance:
            _fail(
                "B2_DESC_COLLECTION_PROVENANCE_MISMATCH",
                f"record provenance drifted for {stable_id}",
            )
        plan.append(
            PlannedDescriptorSample(stable_sample_id=stable_id, membership=membership)
        )
        entries.append(
            AcceptedSampleEntry(
                stable_sample_id=stable_id,
                relative_path=relative,
                record_scientific_sha256=str(claimed_record),
                record_file_sha256=str(claimed_file),
                membership=membership,
            )
        )

    if membership_counts != dict(config.required_split_counts):
        _fail(
            "B2_DESC_CACHE_SPLIT_COUNT_MISMATCH",
            f"membership counts drifted: {membership_counts}",
        )
    if common_provenance is None:
        _fail("B2_DESC_CACHE_RECORD_COUNT_MISMATCH", "teacher-cache samples are empty")
    if (
        common_provenance["candidate_layers"] != tuple(config.candidate_layers)
        or common_provenance["prediction_depths"] != tuple(config.prediction_depths)
        or common_provenance["descriptor_feature_order"] != _FEATURE_ORDER
    ):
        _fail(
            "B2_DESC_COLLECTION_PROVENANCE_MISMATCH",
            "accepted cache provenance does not match tracked configuration pins",
        )
    if (
        common_provenance["checkpoint_sha256"] != config.expected_checkpoint_sha256
        or common_provenance["execution_profile_sha256"] != config.expected_execution_profile_sha256
        or common_provenance["split_scientific_sha256"] != config.expected_split_scientific_sha256
            or common_provenance["descriptor_contract_version"]
            != config.descriptor_contract_version
            or common_provenance["descriptor_extractor_config_sha256"] != expected_cfg
            or common_provenance["descriptor_extractor_implementation_sha256"]
            != expected_impl
    ):
        _fail(
            "B2_DESC_COLLECTION_PROVENANCE_MISMATCH",
            "accepted cache provenance does not match tracked configuration pins",
        )
    recomputed_coverage = cache_mod.recompute_teacher_cache_sample_coverage_sha256(entries)
    recomputed_scientific = cache_mod.recompute_teacher_cache_scientific_sha256(
        verified_entries=entries,
        manifest_contract=manifest,
    )
    if (
        recomputed_coverage != claimed_coverage
        or recomputed_coverage != config.expected_sample_coverage_sha256
    ):
        _fail(
            "B2_DESC_CACHE_COVERAGE_HASH_MISMATCH",
            "teacher-cache sample coverage hash does not reprove from verified entries",
        )
    if (
        recomputed_scientific != claimed_scientific
        or recomputed_scientific != config.expected_teacher_cache_scientific_sha256
    ):
        _fail(
            "B2_DESC_CACHE_SCIENTIFIC_HASH_MISMATCH",
            "teacher-cache scientific hash does not reprove from verified entries",
        )

    return AcceptedTeacherCache(
        manifest=MappingProxyType(dict(manifest)),
        plan=tuple(plan),
        entries=tuple(entries),
    )


def _collection_provenance_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    config: DescriptorArtifactsConfig,
) -> Mapping[str, Any]:
    ordered = sorted(records, key=lambda row: str(row["stable_sample_id"]))
    if not ordered:
        _fail("B2_DESC_COLLECTION_EMPTY", "collection requires records")
    expected_impl, expected_cfg = _authoritative_extractor_digests(
        descriptor_contract_version=config.descriptor_contract_version
    )
    first = ordered[0]
    provenance = {
        "teacher_cache_scientific_sha256": str(first["teacher_cache_scientific_sha256"]),
        "split_scientific_sha256": str(first["split_scientific_sha256"]),
        "checkpoint_sha256": str(first["checkpoint_sha256"]),
        "execution_profile_sha256": str(first["execution_profile_sha256"]),
        "descriptor_contract_version": int(first["descriptor_contract_version"]),
        "descriptor_extractor_config_sha256": str(
            first["descriptor_extractor_config_sha256"]
        ),
        "descriptor_extractor_implementation_sha256": str(
            first["descriptor_extractor_implementation_sha256"]
        ),
        "candidate_layers": tuple(int(value) for value in first["candidate_layers"]),
        "prediction_depths": tuple(int(value) for value in first["prediction_depths"]),
        "descriptor_feature_order": tuple(first["descriptor_feature_order"]),
    }
    for record in ordered[1:]:
        current = {
            "teacher_cache_scientific_sha256": str(record["teacher_cache_scientific_sha256"]),
            "split_scientific_sha256": str(record["split_scientific_sha256"]),
            "checkpoint_sha256": str(record["checkpoint_sha256"]),
            "execution_profile_sha256": str(record["execution_profile_sha256"]),
            "descriptor_contract_version": int(record["descriptor_contract_version"]),
            "descriptor_extractor_config_sha256": str(
                record["descriptor_extractor_config_sha256"]
            ),
            "descriptor_extractor_implementation_sha256": str(
                record["descriptor_extractor_implementation_sha256"]
            ),
            "candidate_layers": tuple(int(value) for value in record["candidate_layers"]),
            "prediction_depths": tuple(int(value) for value in record["prediction_depths"]),
            "descriptor_feature_order": tuple(record["descriptor_feature_order"]),
        }
        if current != provenance:
            _fail(
                "B2_DESC_COLLECTION_PROVENANCE_MISMATCH",
                f"descriptor record provenance drifted for {record['stable_sample_id']}",
            )
    if (
        provenance["teacher_cache_scientific_sha256"]
        != config.expected_teacher_cache_scientific_sha256
        or provenance["split_scientific_sha256"] != config.expected_split_scientific_sha256
        or provenance["checkpoint_sha256"] != config.expected_checkpoint_sha256
        or provenance["execution_profile_sha256"]
        != config.expected_execution_profile_sha256
        or provenance["descriptor_contract_version"] != config.descriptor_contract_version
        or provenance["descriptor_extractor_config_sha256"] != expected_cfg
        or provenance["descriptor_extractor_implementation_sha256"] != expected_impl
        or provenance["candidate_layers"] != tuple(config.candidate_layers)
        or provenance["prediction_depths"] != tuple(config.prediction_depths)
        or provenance["descriptor_feature_order"] != _FEATURE_ORDER
    ):
        _fail(
            "B2_DESC_COLLECTION_PROVENANCE_MISMATCH",
            "descriptor collection provenance does not match tracked configuration pins",
        )
    return MappingProxyType(provenance)


def reconstruct_descriptor_record(
    *,
    teacher_scientific_record: Mapping[str, Any],
    config: DescriptorArtifactsConfig,
    source_teacher_record_scientific_sha256: str,
    source_teacher_record_file_sha256: str,
    teacher_cache_scientific_sha256: str,
    descriptor_feature_order: Sequence[str],
    descriptor_extractor_config_sha256: str,
    descriptor_extractor_implementation_sha256: str,
) -> dict[str, Any]:
    """Build one descriptor record from a teacher-cache scientific record."""

    if tuple(descriptor_feature_order) != _FEATURE_ORDER:
        _fail("B2_DESC_FEATURE_ORDER_MISMATCH", "descriptor feature order drifted")
    descriptors = cache_mod.reconstruct_persisted_descriptors(teacher_scientific_record)
    descriptor_by_depth: dict[int, torch.Tensor] = {}
    valid_layer_mask_by_depth: dict[int, list[bool]] = {}
    for depth in config.prediction_depths:
        tensor = descriptors[depth]
        if not isinstance(tensor, torch.Tensor):
            _fail("B2_DESC_TENSOR_NONFINITE", f"descriptor at depth {depth} missing")
        value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        expected_shape = _expected_descriptor_shape(
            candidate_layers=config.candidate_layers,
            depth=depth,
            descriptor_dimension=config.descriptor_dimension,
        )
        if tuple(value.shape) != expected_shape:
            _fail(
                "B2_DESC_TENSOR_SHAPE_MISMATCH",
                f"depth {depth} shape {tuple(value.shape)} != {expected_shape}",
            )
        if not bool(torch.isfinite(value).all()):
            _fail("B2_DESC_TENSOR_NONFINITE", f"descriptor at depth {depth} is nonfinite")
        descriptor_by_depth[depth] = value
        layer_count = expected_shape[1]
        valid_layer_mask_by_depth[depth] = [True] * layer_count

    record: dict[str, Any] = {
        "schema_version": 1,
        "stable_sample_id": str(teacher_scientific_record["stable_sample_id"]),
        "split_membership": str(teacher_scientific_record["membership"]),
        "category": str(teacher_scientific_record["category"]),
        "label": int(teacher_scientific_record["image_label"]),
        "anomaly_type": str(teacher_scientific_record["anomaly_type"]),
        "candidate_layers": list(config.candidate_layers),
        "prediction_depths": list(config.prediction_depths),
        "descriptor_contract_version": int(config.descriptor_contract_version),
        "descriptor_feature_order": list(descriptor_feature_order),
        "descriptor_extractor_config_sha256": str(descriptor_extractor_config_sha256),
        "descriptor_extractor_implementation_sha256": str(
            descriptor_extractor_implementation_sha256
        ),
        "descriptor_by_depth": descriptor_by_depth,
        "valid_layer_mask_by_depth": valid_layer_mask_by_depth,
        "source_teacher_record_scientific_sha256": str(
            source_teacher_record_scientific_sha256
        ),
        "source_teacher_record_file_sha256": str(source_teacher_record_file_sha256),
        "teacher_cache_scientific_sha256": str(teacher_cache_scientific_sha256),
        "split_scientific_sha256": str(teacher_scientific_record["split_scientific_sha256"]),
        "checkpoint_sha256": str(teacher_scientific_record["checkpoint_sha256"]),
        "execution_profile_sha256": str(
            teacher_scientific_record["execution_profile_sha256"]
        ),
    }
    record["descriptor_record_scientific_sha256"] = descriptor_record_scientific_sha256(
        record
    )
    return record


def _canonicalize_descriptor_tensor(
    depth: int, tensor: torch.Tensor
) -> dict[str, Any]:
    """Canonical tensor meta for scientific hashing.

    Finiteness is enforced by ``validate_descriptor_record``, not here, so callers
    can attach a scientific digest before the fail-closed finiteness gate runs.
    """

    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    name = f"descriptor:{depth}"
    if bool(torch.isfinite(value).all()):
        digest = cache_mod.canonical_tensor_digest(name, value, _DESCRIPTOR_SEMANTICS)
    else:
        # Mirror canonical_tensor_digest payload encoding without the finiteness gate.
        payload = cache_mod._encode_canonical_tensor_payload(  # noqa: SLF001
            name,
            "float32",
            tuple(int(size) for size in value.shape),
            _DESCRIPTOR_SEMANTICS,
            cache_mod._little_endian_contiguous_bytes(value),  # noqa: SLF001
        )
        digest = hashlib.sha256(payload).hexdigest()
    return {
        "dtype": "float32",
        "shape": list(value.shape),
        "dimension_semantics": list(_DESCRIPTOR_SEMANTICS),
        "digest": digest,
    }


def descriptor_record_scientific_content(
    record: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Explicit scientific whitelist for one descriptor record (excludes self-hash)."""

    if not isinstance(record, Mapping):
        _fail("B2_DESC_RECORD_HASH_SCHEMA_INVALID", "record must be an object")
    descriptor_by_depth = record.get("descriptor_by_depth")
    if not isinstance(descriptor_by_depth, Mapping):
        _fail("B2_DESC_RECORD_HASH_SCHEMA_INVALID", "descriptor_by_depth required")
    masks = record.get("valid_layer_mask_by_depth")
    if not isinstance(masks, Mapping):
        _fail("B2_DESC_RECORD_HASH_SCHEMA_INVALID", "valid_layer_mask_by_depth required")

    depth_payload = {
        str(int(depth)): _canonicalize_descriptor_tensor(int(depth), tensor)
        for depth, tensor in sorted(
            ((int(key), value) for key, value in descriptor_by_depth.items()),
            key=lambda item: item[0],
        )
    }
    mask_payload = {
        str(int(depth)): [bool(flag) for flag in masks[depth]]
        for depth in sorted(masks, key=lambda key: int(key))
    }
    content: dict[str, Any] = {
        "schema_version": record["schema_version"],
        "stable_sample_id": record["stable_sample_id"],
        "split_membership": record["split_membership"],
        "category": record["category"],
        "label": record["label"],
        "anomaly_type": record["anomaly_type"],
        "candidate_layers": list(record["candidate_layers"]),
        "prediction_depths": list(record["prediction_depths"]),
        "descriptor_contract_version": record["descriptor_contract_version"],
        "descriptor_feature_order": list(record["descriptor_feature_order"]),
        "descriptor_extractor_config_sha256": record[
            "descriptor_extractor_config_sha256"
        ],
        "descriptor_extractor_implementation_sha256": record[
            "descriptor_extractor_implementation_sha256"
        ],
        "descriptor_by_depth": depth_payload,
        "valid_layer_mask_by_depth": mask_payload,
        "source_teacher_record_scientific_sha256": record[
            "source_teacher_record_scientific_sha256"
        ],
        "teacher_cache_scientific_sha256": record["teacher_cache_scientific_sha256"],
        "split_scientific_sha256": record["split_scientific_sha256"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "execution_profile_sha256": record["execution_profile_sha256"],
    }
    missing = [key for key in _DESCRIPTOR_RECORD_SCIENTIFIC_FIELDS if key not in content]
    if missing:
        _fail(
            "B2_DESC_RECORD_HASH_SCHEMA_INVALID",
            f"missing scientific fields: {missing}",
        )
    return MappingProxyType(content)


def descriptor_record_scientific_sha256(record: Mapping[str, Any]) -> str:
    """SHA-256 over the descriptor-record scientific whitelist."""

    return _canonical_sha256(dict(descriptor_record_scientific_content(record)))


def validate_descriptor_record(
    record: Mapping[str, Any],
    *,
    config: DescriptorArtifactsConfig,
) -> None:
    """Fail-closed validation of one descriptor record."""

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
        if field not in record:
            _fail("B2_DESC_RECORD_SCHEMA_INVALID", f"missing field {field}")

    if tuple(record["candidate_layers"]) != tuple(config.candidate_layers):
        _fail("B2_DESC_RECORD_SCHEMA_INVALID", "candidate_layers drifted")
    if tuple(record["prediction_depths"]) != tuple(config.prediction_depths):
        _fail("B2_DESC_RECORD_SCHEMA_INVALID", "prediction_depths drifted")
    if tuple(record["descriptor_feature_order"]) != _FEATURE_ORDER:
        _fail("B2_DESC_FEATURE_ORDER_MISMATCH", "descriptor feature order drifted")
    if int(record["descriptor_contract_version"]) != config.descriptor_contract_version:
        _fail("B2_DESC_CONTRACT_IDENTITY_MISMATCH", "descriptor contract version drifted")

    descriptor_by_depth = record["descriptor_by_depth"]
    masks = record["valid_layer_mask_by_depth"]
    if not isinstance(descriptor_by_depth, Mapping) or not isinstance(masks, Mapping):
        _fail("B2_DESC_RECORD_SCHEMA_INVALID", "descriptor tensors/masks invalid")
    for depth in config.prediction_depths:
        if depth not in descriptor_by_depth:
            _fail("B2_DESC_DEPTH_MISSING", f"missing descriptor depth {depth}")
        tensor = descriptor_by_depth[depth]
        if not isinstance(tensor, torch.Tensor):
            _fail("B2_DESC_RECORD_SCHEMA_INVALID", f"descriptor depth {depth} not a tensor")
        expected_shape = _expected_descriptor_shape(
            candidate_layers=config.candidate_layers,
            depth=depth,
            descriptor_dimension=config.descriptor_dimension,
        )
        if tuple(tensor.shape) != expected_shape:
            _fail(
                "B2_DESC_TENSOR_SHAPE_MISMATCH",
                f"depth {depth} shape {tuple(tensor.shape)} != {expected_shape}",
            )
        if tensor.dtype != torch.float32:
            _fail("B2_DESC_TENSOR_DTYPE_MISMATCH", f"depth {depth} must be float32")
        if not bool(torch.isfinite(tensor).all()):
            _fail("B2_DESC_TENSOR_NONFINITE", f"depth {depth} contains NaN or Inf")
        mask = masks.get(depth)
        if not isinstance(mask, list | tuple) or len(mask) != expected_shape[1]:
            _fail(
                "B2_DESC_RECORD_SCHEMA_INVALID",
                f"valid_layer_mask_by_depth[{depth}] length mismatch",
            )

    claimed = record["descriptor_record_scientific_sha256"]
    recomputed = descriptor_record_scientific_sha256(record)
    if claimed != recomputed:
        _fail(
            "B2_DESC_RECORD_HASH_MISMATCH",
            "descriptor_record_scientific_sha256 does not match content",
        )


def _validate_descriptor_record_for_statistics(
    record: Mapping[str, Any],
    *,
    config: DescriptorArtifactsConfig,
) -> None:
    """Validate tensors used for statistics without requiring a fresh scientific digest."""

    if tuple(record.get("candidate_layers", ())) != tuple(config.candidate_layers):
        _fail("B2_DESC_NORMALIZATION_SCHEMA_INVALID", "candidate_layers drifted")
    if tuple(record.get("prediction_depths", ())) != tuple(config.prediction_depths):
        _fail("B2_DESC_NORMALIZATION_SCHEMA_INVALID", "prediction_depths drifted")
    descriptor_by_depth = record.get("descriptor_by_depth")
    if not isinstance(descriptor_by_depth, Mapping):
        _fail("B2_DESC_NORMALIZATION_SCHEMA_INVALID", "descriptor_by_depth required")
    for depth in config.prediction_depths:
        if depth not in descriptor_by_depth:
            _fail("B2_DESC_DEPTH_MISSING", f"missing descriptor depth {depth}")
        tensor = descriptor_by_depth[depth]
        if not isinstance(tensor, torch.Tensor):
            _fail("B2_DESC_NORMALIZATION_SCHEMA_INVALID", f"depth {depth} not a tensor")
        expected_shape = _expected_descriptor_shape(
            candidate_layers=config.candidate_layers,
            depth=depth,
            descriptor_dimension=config.descriptor_dimension,
        )
        if tuple(tensor.shape) != expected_shape:
            _fail(
                "B2_DESC_TENSOR_SHAPE_MISMATCH",
                f"depth {depth} shape {tuple(tensor.shape)} != {expected_shape}",
            )
        if tensor.dtype != torch.float32:
            _fail("B2_DESC_TENSOR_DTYPE_MISMATCH", f"depth {depth} must be float32")
        if not bool(torch.isfinite(tensor).all()):
            _fail("B2_DESC_TENSOR_NONFINITE", f"depth {depth} contains NaN or Inf")


def training_sample_coverage_sha256(stable_sample_ids: Sequence[str]) -> str:
    """Order-invariant coverage hash over the exact training ID set."""

    ordered = sorted(str(item) for item in stable_sample_ids)
    return _canonical_sha256({"planned_stable_sample_ids": ordered})


def normalization_training_coverage_sha256(
    *,
    ordered_training_stable_sample_ids: Sequence[str],
    descriptor_record_scientific_sha256_by_id: Mapping[str, str],
) -> str:
    """Bind the exact ordered 16 training IDs to their descriptor-record hashes."""

    ordered = [str(item) for item in ordered_training_stable_sample_ids]
    if ordered != sorted(ordered):
        _fail(
            "B2_DESC_NORMALIZATION_COVERAGE_ORDER",
            "training coverage IDs must be ascending",
        )
    return _canonical_sha256(
        {
            "ordered_training_stable_sample_ids": ordered,
            "descriptor_record_scientific_sha256_by_id": {
                stable_id: descriptor_record_scientific_sha256_by_id[stable_id]
                for stable_id in ordered
            },
        }
    )


def descriptor_sample_coverage_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Bind the exact ordered 32 IDs and 16/8/8 memberships."""

    ordered = sorted(records, key=lambda row: str(row["stable_sample_id"]))
    payload = {
        "ordered_stable_sample_ids": [str(row["stable_sample_id"]) for row in ordered],
        "split_membership_by_id": {
            str(row["stable_sample_id"]): str(row["split_membership"]) for row in ordered
        },
        "membership_counts": {
            "training": sum(1 for row in ordered if row["split_membership"] == "training"),
            "calibration": sum(
                1 for row in ordered if row["split_membership"] == "calibration"
            ),
            "evaluation": sum(
                1 for row in ordered if row["split_membership"] == "evaluation"
            ),
        },
    }
    return _canonical_sha256(payload)


def descriptor_collection_scientific_content(
    *,
    records: Sequence[Mapping[str, Any]],
    statistics: Mapping[str, Any],
    config: DescriptorArtifactsConfig,
) -> Mapping[str, Any]:
    """Explicit whitelist for the 32-sample descriptor collection identity."""

    ordered = sorted(records, key=lambda row: str(row["stable_sample_id"]))
    provenance = _collection_provenance_from_records(ordered, config=config)
    content = {
        "ordered_stable_sample_ids": [str(row["stable_sample_id"]) for row in ordered],
        "split_membership_by_id": {
            str(row["stable_sample_id"]): str(row["split_membership"]) for row in ordered
        },
        "descriptor_record_scientific_sha256_by_id": {
            str(row["stable_sample_id"]): str(row["descriptor_record_scientific_sha256"])
            for row in ordered
        },
        "candidate_layers": list(config.candidate_layers),
        "prediction_depths": list(config.prediction_depths),
        "descriptor_feature_order": list(_FEATURE_ORDER),
        "descriptor_contract_version": int(config.descriptor_contract_version),
        "descriptor_extractor_config_sha256": str(
            provenance["descriptor_extractor_config_sha256"]
        ),
        "descriptor_extractor_implementation_sha256": str(
            provenance["descriptor_extractor_implementation_sha256"]
        ),
        "teacher_cache_scientific_sha256": str(
            provenance["teacher_cache_scientific_sha256"]
        ),
        "split_scientific_sha256": str(provenance["split_scientific_sha256"]),
        "checkpoint_sha256": str(provenance["checkpoint_sha256"]),
        "execution_profile_sha256": str(provenance["execution_profile_sha256"]),
        "normalization_statistics_scientific_sha256": str(
            statistics["normalization_statistics_scientific_sha256"]
        ),
    }
    return MappingProxyType(content)


def descriptor_collection_scientific_sha256(
    *,
    records: Sequence[Mapping[str, Any]],
    statistics: Mapping[str, Any],
    config: DescriptorArtifactsConfig,
) -> str:
    """SHA-256 over the collection scientific whitelist."""

    return _canonical_sha256(
        dict(
            descriptor_collection_scientific_content(
                records=records, statistics=statistics, config=config
            )
        )
    )


def _feature_stat_payload(
    *,
    descriptor_feature_name: str,
    count: int,
    mean: float,
    std: float,
    minimum: float,
    maximum: float,
    zero_variance: bool,
) -> dict[str, Any]:
    return {
        "descriptor_feature_name": descriptor_feature_name,
        "count": count,
        "mean": float(mean),
        "std": float(std),
        "minimum": float(minimum),
        "maximum": float(maximum),
        "zero_variance": bool(zero_variance),
    }


def _population_stats(values: Sequence[float]) -> dict[str, Any]:
    """Deterministic two-pass float64 population statistics (ddof=0)."""

    count = len(values)
    if count == 0:
        _fail("B2_DESC_NORMALIZATION_EMPTY_AXIS", "no valid values for statistics axis")
    mean = math.fsum(float(value) for value in values) / float(count)
    variance = math.fsum((float(value) - mean) ** 2 for value in values) / float(count)
    std = math.sqrt(variance)
    minimum = float(min(values))
    maximum = float(max(values))
    zero_variance = std == 0.0
    return {
        "count": count,
        "mean": float(mean),
        "std": float(0.0 if zero_variance else std),
        "minimum": minimum,
        "maximum": maximum,
        "zero_variance": zero_variance,
    }


def compute_training_normalization_statistics(
    records: Sequence[Mapping[str, Any]],
    *,
    config: DescriptorArtifactsConfig,
) -> dict[str, Any]:
    """Compute frozen normalization statistics from exactly the training set.

    Axes are keyed by prediction_depth / candidate_layer_id / descriptor_feature_name.
    Accumulation is deterministic two-pass float64 with population std (ddof=0).
    Only valid-layer mask entries contribute. Raw input records are never mutated.
    """

    expected_training = int(config.required_split_counts["training"])
    if len(records) != expected_training:
        _fail(
            "B2_DESC_NORMALIZATION_COUNT_MISMATCH",
            f"expected exactly {expected_training} training records, got {len(records)}",
        )
    ids: list[str] = []
    for record in records:
        membership = record.get("split_membership")
        if membership != "training":
            _fail(
                "B2_DESC_NORMALIZATION_MEMBERSHIP_INVALID",
                f"non-training membership entered statistics: {membership!r}",
            )
        stable_id = str(record.get("stable_sample_id", ""))
        if not _is_sha256(stable_id):
            _fail(
                "B2_DESC_NORMALIZATION_SAMPLE_INVALID",
                f"invalid training stable_sample_id: {stable_id!r}",
            )
        ids.append(stable_id)
        _validate_descriptor_record_for_statistics(record, config=config)
    if len(set(ids)) != len(ids):
        _fail(
            "B2_DESC_NORMALIZATION_DUPLICATE_SAMPLE",
            "duplicate training stable_sample_id in statistics input",
        )

    ordered_records = sorted(records, key=lambda row: str(row["stable_sample_id"]))
    ordered_ids = [str(row["stable_sample_id"]) for row in ordered_records]
    coverage = training_sample_coverage_sha256(ordered_ids)
    record_hashes = {
        str(row["stable_sample_id"]): str(row["descriptor_record_scientific_sha256"])
        for row in ordered_records
    }
    training_coverage = normalization_training_coverage_sha256(
        ordered_training_stable_sample_ids=ordered_ids,
        descriptor_record_scientific_sha256_by_id=record_hashes,
    )

    axes: dict[int, Any] = {}
    for depth in config.prediction_depths:
        layer_ids = _layers_for_depth(config.candidate_layers, depth)
        layer_stats: list[dict[str, Any]] = []
        for layer_index, layer_id in enumerate(layer_ids):
            values_by_feature: list[list[float]] = [
                [] for _ in range(config.descriptor_dimension)
            ]
            for row in ordered_records:
                mask = row["valid_layer_mask_by_depth"][depth]
                if not bool(mask[layer_index]):
                    continue
                tensor = row["descriptor_by_depth"][depth]
                for feature_index in range(config.descriptor_dimension):
                    values_by_feature[feature_index].append(
                        float(tensor[0, layer_index, feature_index].item())
                    )
            if all(len(values) == 0 for values in values_by_feature):
                continue
            feature_stats = []
            for feature_index, feature_name in enumerate(_FEATURE_ORDER):
                stats = _population_stats(values_by_feature[feature_index])
                feature_stats.append(
                    _feature_stat_payload(
                        descriptor_feature_name=feature_name,
                        **stats,
                    )
                )
            layer_stats.append(
                {
                    "candidate_layer_position": layer_index,
                    "candidate_layer_id": int(layer_id),
                    "features": feature_stats,
                }
            )
        axes[depth] = {"prediction_depth": int(depth), "layers": layer_stats}

    statistics: dict[str, Any] = {
        "normalization_contract_version": config.normalization_contract_version,
        "training_sample_coverage_sha256": coverage,
        "normalization_training_coverage_sha256": training_coverage,
        "ordered_training_stable_sample_ids": ordered_ids,
        "descriptor_record_scientific_sha256_by_id": record_hashes,
        "axes": axes,
        "statistics_dtype": "float64",
        "application_output_dtype": "float32",
        "standard_deviation_ddof": 0,
    }
    statistics["normalization_statistics_scientific_sha256"] = (
        normalization_statistics_scientific_sha256(statistics)
    )
    return statistics


def normalization_statistics_scientific_content(
    statistics: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Explicit whitelist for normalization scientific identity."""

    axes_raw = statistics.get("axes")
    if not isinstance(axes_raw, Mapping):
        _fail("B2_DESC_NORMALIZATION_SCHEMA_INVALID", "axes mapping required")
    axes: dict[str, Any] = {}
    for depth in sorted(axes_raw, key=lambda key: int(key)):
        depth_payload = axes_raw[depth]
        layers = depth_payload["layers"]
        axes[str(int(depth))] = {
            "prediction_depth": int(depth),
            "layers": [
                {
                    "candidate_layer_position": int(layer["candidate_layer_position"]),
                    "candidate_layer_id": int(layer["candidate_layer_id"]),
                    "features": [
                        _feature_stat_payload(
                            descriptor_feature_name=str(
                                feature["descriptor_feature_name"]
                            ),
                            count=int(feature["count"]),
                            mean=float(feature["mean"]),
                            std=float(feature["std"]),
                            minimum=float(feature["minimum"]),
                            maximum=float(feature["maximum"]),
                            zero_variance=bool(feature["zero_variance"]),
                        )
                        for feature in layer["features"]
                    ],
                }
                for layer in layers
            ],
        }
    content = {
        "normalization_contract_version": statistics["normalization_contract_version"],
        "ordered_training_stable_sample_ids": list(
            statistics["ordered_training_stable_sample_ids"]
        ),
        "descriptor_record_scientific_sha256_by_id": dict(
            statistics["descriptor_record_scientific_sha256_by_id"]
        ),
        "axes": axes,
        "training_sample_coverage_sha256": statistics["training_sample_coverage_sha256"],
        "normalization_training_coverage_sha256": statistics[
            "normalization_training_coverage_sha256"
        ],
        "statistics_dtype": statistics["statistics_dtype"],
        "standard_deviation_ddof": int(statistics["standard_deviation_ddof"]),
    }
    missing = [key for key in _NORMALIZATION_SCIENTIFIC_FIELDS if key not in content]
    if missing:
        _fail(
            "B2_DESC_NORMALIZATION_SCHEMA_INVALID",
            f"missing normalization scientific fields: {missing}",
        )
    return MappingProxyType(content)


def normalization_statistics_scientific_sha256(statistics: Mapping[str, Any]) -> str:
    """SHA-256 over the normalization scientific whitelist."""

    return _canonical_sha256(dict(normalization_statistics_scientific_content(statistics)))


def apply_frozen_normalization(
    record: Mapping[str, Any],
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply frozen training statistics without mutating the input record.

    Arithmetic runs in float64; normalized tensors are cast to float32.
    Zero-variance rule: ``std == 0`` → ``zero_variance=true``, divisor ``1.0``,
    normalized value ``0``. Layer identity is authoritative via ``candidate_layer_id``.
    """

    axes = statistics["axes"]
    normalized = dict(record)
    descriptor_by_depth: dict[Any, torch.Tensor] = {}
    for depth, tensor in record["descriptor_by_depth"].items():
        depth_key = depth if depth in axes else int(depth)
        depth_stats = axes[depth_key]
        source = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        output = torch.empty_like(source)
        layer_by_id = {
            int(layer["candidate_layer_id"]): layer for layer in depth_stats["layers"]
        }
        candidate_layers = tuple(int(value) for value in record["candidate_layers"])
        layer_ids = _layers_for_depth(candidate_layers, int(depth))
        for layer_index, layer_id in enumerate(layer_ids):
            layer_stats = layer_by_id.get(int(layer_id))
            if layer_stats is None:
                output[0, layer_index, :] = source[0, layer_index, :]
                continue
            for feature_index, feature_stats in enumerate(layer_stats["features"]):
                mean = float(feature_stats["mean"])
                std = float(feature_stats["std"])
                zero_variance = bool(feature_stats["zero_variance"]) or std == 0.0
                raw = float(source[0, layer_index, feature_index].item())
                if zero_variance:
                    value = 0.0
                else:
                    value = (raw - mean) / std
                output[0, layer_index, feature_index] = float(value)
        output = output.to(dtype=torch.float32)
        if not bool(torch.isfinite(output).all()):
            _fail(
                "B2_DESC_NORMALIZATION_NONFINITE",
                f"normalized descriptor at depth {depth} is nonfinite",
            )
        descriptor_by_depth[depth] = output
    normalized["descriptor_by_depth"] = descriptor_by_depth
    return normalized


def descriptor_relative_path(stable_sample_id: str) -> str:
    if not _is_sha256(stable_sample_id):
        _fail("B2_DESC_SAMPLE_ID_INVALID", "stable_sample_id must be sha256 hex")
    return f"descriptors/{stable_sample_id}.pt"


def write_descriptor_record_atomic(
    destination: Path | str,
    record: Mapping[str, Any],
    *,
    config: DescriptorArtifactsConfig,
) -> PersistedDescriptorEntry:
    """Atomically persist one descriptor record; file hash is computed after write."""

    path = Path(destination)
    stable_id = str(record["stable_sample_id"])
    if path.name != f"{stable_id}.pt":
        _fail(
            "B2_DESC_SAMPLE_IDENTITY_INVALID",
            "destination filename must equal the stable sample ID",
        )
    if path.exists():
        _fail("B2_DESC_OVERWRITE_FORBIDDEN", f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    scientific_digest = str(record["descriptor_record_scientific_sha256"])
    if scientific_digest != descriptor_record_scientific_sha256(record):
        _fail(
            "B2_DESC_RECORD_HASH_MISMATCH",
            "descriptor_record_scientific_sha256 does not match content before write",
        )
    # File hash must never enter the hashed payload.
    persistable = {
        key: value
        for key, value in dict(record).items()
        if key != "descriptor_record_file_sha256"
    }
    payload = {
        "scientific_record": persistable,
        "descriptor_record_scientific_sha256": scientific_digest,
    }
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    file_digest = _sha256_file(path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, Mapping) or set(loaded) != {
        "scientific_record",
        "descriptor_record_scientific_sha256",
    }:
        _fail("B2_DESC_PT_PAYLOAD_INVALID", "descriptor .pt payload keys are not exact")
    if "descriptor_record_file_sha256" in loaded["scientific_record"]:
        _fail(
            "B2_DESC_FILE_HASH_IN_PAYLOAD",
            "descriptor_record_file_sha256 must not be stored inside the .pt payload",
        )
    embedded = loaded["descriptor_record_scientific_sha256"]
    recomputed = descriptor_record_scientific_sha256(loaded["scientific_record"])
    if embedded != scientific_digest or recomputed != scientific_digest or embedded != recomputed:
        _fail(
            "B2_DESC_RECORD_HASH_MISMATCH",
            "descriptor scientific hash drifted after reload",
        )
    validate_descriptor_record(loaded["scientific_record"], config=config)
    return PersistedDescriptorEntry(
        stable_sample_id=stable_id,
        relative_record_path=descriptor_relative_path(stable_id),
        descriptor_record_scientific_sha256=scientific_digest,
        descriptor_record_file_sha256=file_digest,
        verification_status="verified",
    )


def _load_descriptor_payload(path: Path) -> Mapping[str, Any]:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, Mapping):
        _fail("B2_DESC_DIGEST_ONLY_RECORD", f"invalid descriptor payload at {path}")
    if "descriptor_by_depth" in loaded:
        return loaded
    record = loaded.get("scientific_record")
    if isinstance(record, Mapping) and "descriptor_by_depth" in record:
        return record
    # Legacy key used only by pre-gap-1 drafts; still accept for migration checks.
    legacy = loaded.get("descriptor_record")
    if isinstance(legacy, Mapping) and "descriptor_by_depth" in legacy:
        return legacy
    _fail(
        "B2_DESC_DIGEST_ONLY_RECORD",
        f"descriptor payload missing tensor values at {path}",
    )


def verify_persisted_descriptor_entry(
    *,
    run_dir: Path | str,
    entry: PersistedDescriptorEntry,
    config: DescriptorArtifactsConfig,
) -> Mapping[str, Any]:
    """Resume/finalization verification for one descriptor artifact."""

    root = Path(run_dir).resolve()
    normalized_relative = _normalized_relative_path(
        entry.relative_record_path,
        code="B2_DESC_RUN_RELATIVE_PATH_INVALID",
    )
    if normalized_relative != descriptor_relative_path(entry.stable_sample_id):
        _fail(
            "B2_DESC_RUN_RELATIVE_PATH_INVALID",
            f"descriptor path drifted for {entry.stable_sample_id}",
        )
    path = resolve_run_relative_artifact(
        run_dir=root,
        relative_path=normalized_relative,
        expected_kind=f"descriptor record {entry.stable_sample_id}",
    )
    file_digest = _sha256_file(path)
    if file_digest != entry.descriptor_record_file_sha256:
        _fail(
            "B2_DESC_RECORD_FILE_HASH_MISMATCH",
            f"descriptor file hash mismatch for {entry.stable_sample_id}",
        )
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, Mapping) or set(loaded) != {
        "scientific_record",
        "descriptor_record_scientific_sha256",
    }:
        _fail("B2_DESC_PT_PAYLOAD_INVALID", "descriptor .pt payload keys are not exact")
    if "descriptor_record_file_sha256" in loaded["scientific_record"]:
        _fail(
            "B2_DESC_FILE_HASH_IN_PAYLOAD",
            "descriptor_record_file_sha256 must not be stored inside the .pt payload",
        )
    scientific_record = loaded["scientific_record"]
    embedded = loaded["descriptor_record_scientific_sha256"]
    recomputed = descriptor_record_scientific_sha256(scientific_record)
    if (
        embedded != entry.descriptor_record_scientific_sha256
        or recomputed != entry.descriptor_record_scientific_sha256
    ):
        _fail(
            "B2_DESC_RECORD_HASH_MISMATCH",
            f"descriptor scientific hash mismatch for {entry.stable_sample_id}",
        )
    validate_descriptor_record(scientific_record, config=config)
    return scientific_record


def write_normalization_statistics_atomic(
    destination: Path | str,
    statistics: Mapping[str, Any],
) -> PersistedNormalizationEntry:
    """Persist frozen normalization statistics; file hash computed after write."""

    path = Path(destination)
    if path.exists():
        _fail("B2_DESC_OVERWRITE_FORBIDDEN", f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    scientific = dict(statistics)
    scientific.pop("normalization_statistics_file_sha256", None)
    digest = normalization_statistics_scientific_sha256(scientific)
    if digest != statistics.get("normalization_statistics_scientific_sha256"):
        _fail(
            "B2_DESC_NORMALIZATION_HASH_MISMATCH",
            "normalization scientific hash drifted before write",
        )
    payload = {
        "scientific_statistics_record": scientific,
        "normalization_statistics_scientific_sha256": digest,
    }
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    file_digest = _sha256_file(path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, Mapping) or set(loaded) != {
        "scientific_statistics_record",
        "normalization_statistics_scientific_sha256",
    }:
        _fail("B2_DESC_PT_PAYLOAD_INVALID", "normalization payload keys are not exact")
    embedded = loaded["normalization_statistics_scientific_sha256"]
    recomputed = normalization_statistics_scientific_sha256(
        loaded["scientific_statistics_record"]
    )
    if embedded != digest or recomputed != digest or embedded != recomputed:
        _fail(
            "B2_DESC_NORMALIZATION_HASH_MISMATCH",
            "normalization scientific hash drifted after reload",
        )
    relative = path.name if path.parent.name != "descriptors" else path.name
    if path.name == Path(_NORMALIZATION_RELATIVE_PATH).name:
        relative = _NORMALIZATION_RELATIVE_PATH
    return PersistedNormalizationEntry(
        relative_path=relative,
        normalization_statistics_scientific_sha256=digest,
        normalization_statistics_file_sha256=file_digest,
    )


def verify_persisted_normalization_entry(
    *,
    run_dir: Path | str,
    entry: PersistedNormalizationEntry,
    config: DescriptorArtifactsConfig | None = None,
) -> Mapping[str, Any]:
    root = Path(run_dir).resolve()
    normalized_relative = _normalized_relative_path(
        entry.relative_path,
        code="B2_DESC_RUN_RELATIVE_PATH_INVALID",
    )
    if normalized_relative != _NORMALIZATION_RELATIVE_PATH:
        _fail(
            "B2_DESC_RUN_RELATIVE_PATH_INVALID",
            "normalization path must equal the tracked run-relative artifact path",
        )
    path = resolve_run_relative_artifact(
        run_dir=root,
        relative_path=normalized_relative,
        expected_kind="normalization statistics artifact",
    )
    file_digest = _sha256_file(path)
    if file_digest != entry.normalization_statistics_file_sha256:
        _fail(
            "B2_DESC_NORMALIZATION_FILE_HASH_MISMATCH",
            "normalization statistics file hash mismatch",
        )
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, Mapping) or set(loaded) != {
        "scientific_statistics_record",
        "normalization_statistics_scientific_sha256",
    }:
        _fail("B2_DESC_PT_PAYLOAD_INVALID", "normalization payload keys are not exact")
    scientific = loaded["scientific_statistics_record"]
    if "normalization_statistics_file_sha256" in scientific:
        _fail(
            "B2_DESC_FILE_HASH_IN_PAYLOAD",
            "normalization file hash must not be stored inside the payload",
        )
    recomputed = normalization_statistics_scientific_sha256(scientific)
    if (
        loaded["normalization_statistics_scientific_sha256"]
        != entry.normalization_statistics_scientific_sha256
        or recomputed != entry.normalization_statistics_scientific_sha256
    ):
        _fail(
            "B2_DESC_NORMALIZATION_HASH_MISMATCH",
            "normalization scientific hash mismatch after reload",
        )
    if config is not None:
        if (
            int(scientific["normalization_contract_version"])
            != config.normalization_contract_version
        ):
            _fail(
                "B2_DESC_NORMALIZATION_SCHEMA_INVALID",
                "normalization contract version drifted",
            )
        expected_training = int(config.required_split_counts["training"])
        ordered_training_ids = scientific["ordered_training_stable_sample_ids"]
        if len(ordered_training_ids) != expected_training:
            _fail(
                "B2_DESC_NORMALIZATION_COUNT_MISMATCH",
                "normalization training coverage is incomplete",
            )
    return scientific


def write_final_manifest_with_receipt_atomic(
    run_dir: Path | str,
    manifest: Mapping[str, Any],
) -> str:
    """Atomically write final_manifest.json and its non-self-referential SHA receipt."""

    from rad.artifacts import atomic_write_json

    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / _FINAL_MANIFEST_NAME
    receipt_path = root / _FINAL_MANIFEST_RECEIPT_NAME
    if manifest_path.exists() or receipt_path.exists():
        _fail("B2_DESC_OVERWRITE_FORBIDDEN", "final manifest or receipt already exists")
    # Manifest must not embed its own file hash.
    payload = {
        key: value
        for key, value in dict(manifest).items()
        if key != "final_manifest_file_sha256"
    }
    atomic_write_json(manifest_path, payload)
    digest = _sha256_file(manifest_path)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{receipt_path.name}.",
        suffix=".tmp",
        dir=root,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(digest + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, receipt_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return digest


def verify_final_manifest_receipt(run_dir: Path | str) -> str:
    root = Path(run_dir).resolve()
    manifest_path = resolve_run_relative_artifact(
        run_dir=root,
        relative_path=_FINAL_MANIFEST_NAME,
        expected_kind="final manifest",
    )
    receipt_path = resolve_run_relative_artifact(
        run_dir=root,
        relative_path=_FINAL_MANIFEST_RECEIPT_NAME,
        expected_kind="final manifest receipt",
    )
    actual = _sha256_file(manifest_path)
    claimed = receipt_path.read_text(encoding="utf-8").strip()
    if claimed != actual:
        _fail(
            "B2_DESC_MANIFEST_RECEIPT_MISMATCH",
            "final_manifest.json.sha256 does not match manifest bytes",
        )
    return actual


def audit_descriptor_artifact_integrity(
    *,
    run_dir: Path | str,
    manifest: Mapping[str, Any],
    planned_ids: Sequence[str],
    config: DescriptorArtifactsConfig | None = None,
) -> None:
    """Fail-closed audit of on-disk descriptor artifacts versus the planned set."""

    root = Path(run_dir)
    descriptors_dir = root / "descriptors"
    planned = [str(item) for item in planned_ids]
    planned_set = set(planned)
    if len(planned_set) != len(planned):
        _fail("B2_DESC_PLAN_DUPLICATE", "planned_ids contain duplicates")

    on_disk: dict[str, Path] = {}
    if descriptors_dir.is_dir():
        for path in sorted(descriptors_dir.glob("*.pt")):
            stable_id = path.stem
            if stable_id not in planned_set:
                _fail(
                    "B2_DESC_ORPHAN_ARTIFACT",
                    f"orphan descriptor artifact: {path.name}",
                )
            on_disk[stable_id] = path

    missing = [stable_id for stable_id in planned if stable_id not in on_disk]
    status = manifest.get("status")
    has_normalization = (
        "normalization_statistics" in manifest
        or "normalization_statistics_relative_path" in manifest
    )

    if missing:
        if has_normalization:
            _fail(
                "B2_DESC_NORMALIZATION_BEFORE_COVERAGE",
                "normalization present before complete training coverage",
            )
        if status == "passed":
            _fail(
                "B2_DESC_PARTIAL_CLAIMING_PASSED",
                f"manifest claims passed but missing {len(missing)} descriptor records",
            )
        _fail(
            "B2_DESC_MISSING_ARTIFACT",
            f"missing descriptor records: {missing[:3]}",
        )

    samples = manifest.get("samples")
    if isinstance(samples, list) and samples and config is not None:
        for raw in samples:
            entry = PersistedDescriptorEntry(
                stable_sample_id=str(raw["stable_sample_id"]),
                relative_record_path=str(
                    raw.get("relative_record_path")
                    or descriptor_relative_path(str(raw["stable_sample_id"]))
                ),
                descriptor_record_scientific_sha256=str(
                    raw["descriptor_record_scientific_sha256"]
                ),
                descriptor_record_file_sha256=str(raw["descriptor_record_file_sha256"]),
                verification_status=str(raw.get("verification_status", "verified")),
            )
            verify_persisted_descriptor_entry(
                run_dir=root, entry=entry, config=config
            )
    else:
        for _stable_id, path in on_disk.items():
            _load_descriptor_payload(path)

    if has_normalization and len(on_disk) < len(planned_set):
        _fail(
            "B2_DESC_NORMALIZATION_BEFORE_COVERAGE",
            "normalization generated before complete training coverage",
        )


def build_descriptor_extraction_plan(
    *,
    accepted: AcceptedTeacherCache,
    config: DescriptorArtifactsConfig,
) -> dict[str, Any]:
    """Summarize the accepted cache into a descriptor extraction / normalization plan."""

    membership_counts = {"training": 0, "calibration": 0, "evaluation": 0}
    for row in accepted.plan:
        if row.membership not in membership_counts:
            _fail(
                "B2_DESC_MEMBERSHIP_INVALID",
                f"unexpected membership in accepted plan: {row.membership!r}",
            )
        membership_counts[row.membership] += 1
    if membership_counts != dict(config.required_split_counts):
        _fail(
            "B2_DESC_CACHE_RECORD_COUNT_MISMATCH",
            f"accepted membership counts drifted: {membership_counts}",
        )
    ordered_plan = sorted(accepted.plan, key=lambda row: row.stable_sample_id)
    return {
        "planned_samples": len(accepted.plan),
        "training_samples_for_normalization": membership_counts["training"],
        "calibration_samples_for_normalization": 0,
        "evaluation_samples_for_normalization": 0,
        "prediction_depths": list(config.prediction_depths),
        "candidate_layers": list(config.candidate_layers),
        "descriptor_dimension": config.descriptor_dimension,
        "teacher_forward_count": 0,
        "planned_ordered_stable_sample_ids": [
            row.stable_sample_id for row in ordered_plan
        ],
        "planned_split_membership_by_id": {
            row.stable_sample_id: row.membership for row in ordered_plan
        },
        "planned_training_stable_sample_ids": sorted(
            row.stable_sample_id
            for row in accepted.plan
            if row.membership == "training"
        ),
        "intended_descriptor_sample_coverage_inputs": {
            "ordered_stable_sample_ids": [row.stable_sample_id for row in ordered_plan],
            "split_membership_by_id": {
                row.stable_sample_id: row.membership for row in ordered_plan
            },
            "membership_counts": dict(membership_counts),
        },
        "intended_normalization_training_coverage_inputs": {
            "ordered_training_stable_sample_ids": sorted(
                row.stable_sample_id
                for row in accepted.plan
                if row.membership == "training"
            ),
            "descriptor_record_scientific_sha256_by_id": {},
        },
    }


def build_descriptor_artifacts_manifest(
    *,
    config: DescriptorArtifactsConfig,
    records: Sequence[Mapping[str, Any]],
    statistics: Mapping[str, Any],
    sample_entries: Sequence[PersistedDescriptorEntry] | None = None,
    normalization_entry: PersistedNormalizationEntry | None = None,
) -> dict[str, Any]:
    """Build a passed descriptor-artifacts manifest from complete records + stats."""

    if len(records) != sum(config.required_split_counts.values()):
        _fail(
            "B2_DESC_CACHE_RECORD_COUNT_MISMATCH",
            "manifest requires the complete 32-sample descriptor set",
        )
    for record in records:
        validate_descriptor_record(record, config=config)
    ordered = sorted(records, key=lambda item: str(item["stable_sample_id"]))
    ordered_ids = [str(row["stable_sample_id"]) for row in ordered]
    if sample_entries is None or normalization_entry is None:
        _fail(
            "B2_DESC_PASSED_MANIFEST_REQUIRES_VERIFIED_DISK",
            "passed manifest requires complete verified descriptor and normalization entries",
        )
    if len(sample_entries) != len(ordered):
        _fail(
            "B2_DESC_PASSED_MANIFEST_REQUIRES_VERIFIED_DISK",
            "passed manifest requires one verified entry per descriptor record",
        )
    for entry in sample_entries:
        if (
            entry.verification_status != "verified"
            or not entry.descriptor_record_file_sha256
            or not _is_sha256(entry.descriptor_record_file_sha256)
            or not _is_sha256(entry.descriptor_record_scientific_sha256)
        ):
            _fail(
                "B2_DESC_PASSED_MANIFEST_REQUIRES_VERIFIED_DISK",
                "descriptor entries must all be dual-hash verified",
            )
    if (
        not _is_sha256(normalization_entry.normalization_statistics_scientific_sha256)
        or not _is_sha256(normalization_entry.normalization_statistics_file_sha256)
    ):
        _fail(
            "B2_DESC_PASSED_MANIFEST_REQUIRES_VERIFIED_DISK",
            "normalization entry must be dual-hash verified",
        )
    collection = descriptor_collection_scientific_sha256(
        records=ordered, statistics=statistics, config=config
    )
    coverage = descriptor_sample_coverage_sha256(ordered)
    samples_payload: list[dict[str, Any]]
    by_id = {entry.stable_sample_id: entry for entry in sample_entries}
    samples_payload = [
        {
            "stable_sample_id": entry.stable_sample_id,
            "relative_record_path": entry.relative_record_path,
            "descriptor_record_scientific_sha256": entry.descriptor_record_scientific_sha256,
            "descriptor_record_file_sha256": entry.descriptor_record_file_sha256,
            "verification_status": entry.verification_status,
            "split_membership": next(
                row["split_membership"]
                for row in ordered
                if row["stable_sample_id"] == entry.stable_sample_id
            ),
        }
        for entry in (by_id[stable_id] for stable_id in ordered_ids)
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "configuration_id": config.configuration_id,
        "descriptor_dimension": config.descriptor_dimension,
        "candidate_layers": list(config.candidate_layers),
        "prediction_depths": list(config.prediction_depths),
        "descriptor_contract_version": config.descriptor_contract_version,
        "normalization_contract_version": config.normalization_contract_version,
        "planned_stable_sample_ids": ordered_ids,
        "descriptor_collection_scientific_sha256": collection,
        "descriptor_sample_coverage_sha256": coverage,
        "sample_coverage_sha256": coverage,
        "training_sample_coverage_sha256": statistics["training_sample_coverage_sha256"],
        "normalization_training_coverage_sha256": statistics[
            "normalization_training_coverage_sha256"
        ],
        "normalization_statistics_scientific_sha256": statistics[
            "normalization_statistics_scientific_sha256"
        ],
        "expected_teacher_cache_scientific_sha256": (
            config.expected_teacher_cache_scientific_sha256
        ),
        "expected_split_scientific_sha256": config.expected_split_scientific_sha256,
        "expected_checkpoint_sha256": config.expected_checkpoint_sha256,
        "expected_execution_profile_sha256": config.expected_execution_profile_sha256,
        "samples": samples_payload,
    }
    if normalization_entry is not None:
        manifest["normalization_statistics_relative_path"] = normalization_entry.relative_path
        manifest["normalization_statistics_file_sha256"] = normalization_entry.normalization_statistics_file_sha256
    manifest["teacher_forward_count"] = 0
    return manifest


def build_planned_descriptor_artifacts_manifest(
    *,
    accepted: AcceptedTeacherCache,
    config: DescriptorArtifactsConfig,
) -> dict[str, Any]:
    """Build a structurally distinct planned manifest for no-write planning paths."""

    plan = build_descriptor_extraction_plan(accepted=accepted, config=config)
    return {
        "schema_version": 1,
        "status": "planned",
        "configuration_id": config.configuration_id,
        "planned_stable_sample_ids": list(plan["planned_ordered_stable_sample_ids"]),
        "candidate_layers": list(config.candidate_layers),
        "prediction_depths": list(config.prediction_depths),
        "descriptor_dimension": config.descriptor_dimension,
        "teacher_forward_count": 0,
    }


def materialize_descriptor_artifact_collection(
    *,
    config: DescriptorArtifactsConfig,
    teacher_cache_manifest_path: Path,
    teacher_cache_root: Path,
    output_run_dir: Path,
    validated_teacher_cache: ValidatedTeacherCache | None = None,
) -> DescriptorCollectionResult:
    """Materialize and verify a full descriptor artifact collection without teacher calls."""

    authoritative = load_disk_authoritative_teacher_cache_manifest(
        teacher_cache_manifest_path=teacher_cache_manifest_path,
        teacher_cache_root=teacher_cache_root,
    )
    authoritative_accepted = validate_accepted_teacher_cache(
        manifest=authoritative.manifest,
        config=config,
        cache_root=authoritative.cache_root,
        allow_test_fixture=False,
    )
    if validated_teacher_cache is None:
        validated = ValidatedTeacherCache(
            manifest=authoritative.manifest,
            manifest_path=authoritative.manifest_path,
            cache_root=authoritative.cache_root,
            source_teacher_cache_manifest_file_sha256=(
                authoritative.source_teacher_cache_manifest_file_sha256
            ),
            accepted=authoritative_accepted,
        )
    else:
        if (
            validated_teacher_cache.manifest_path != authoritative.manifest_path
            or validated_teacher_cache.cache_root != authoritative.cache_root
            or validated_teacher_cache.source_teacher_cache_manifest_file_sha256
            != authoritative.source_teacher_cache_manifest_file_sha256
            or dict(validated_teacher_cache.manifest) != dict(authoritative.manifest)
            or validated_teacher_cache.accepted.plan != authoritative_accepted.plan
            or validated_teacher_cache.accepted.entries != authoritative_accepted.entries
            or dict(validated_teacher_cache.accepted.manifest)
            != dict(authoritative_accepted.manifest)
        ):
            _fail(
                "B2_DESC_COLLECTION_PROVENANCE_MISMATCH",
                "validated teacher-cache object does not match authoritative disk inputs",
            )
        validated = validated_teacher_cache
    accepted = authoritative_accepted
    output_run_dir.mkdir(parents=True, exist_ok=False)
    (output_run_dir / "descriptors").mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    entries: list[PersistedDescriptorEntry] = []
    for accepted_entry in accepted.entries:
        payload = torch.load(
            validated.cache_root / accepted_entry.relative_path,
            map_location="cpu",
            weights_only=True,
        )
        scientific_record = payload["scientific_record"]
        record = reconstruct_descriptor_record(
            teacher_scientific_record=scientific_record,
            config=config,
            source_teacher_record_scientific_sha256=accepted_entry.record_scientific_sha256,
            source_teacher_record_file_sha256=accepted_entry.record_file_sha256,
            teacher_cache_scientific_sha256=str(
                validated.manifest["cache_scientific_sha256"]
            ),
            descriptor_feature_order=_FEATURE_ORDER,
            descriptor_extractor_config_sha256=scientific_record["extractor_configuration_sha256"],
            descriptor_extractor_implementation_sha256=scientific_record[
                "descriptor_implementation_sha256"
            ],
        )
        records.append(record)
        entries.append(
            write_descriptor_record_atomic(
                output_run_dir / descriptor_relative_path(record["stable_sample_id"]),
                record,
                config=config,
            )
        )
    training_records = [row for row in records if row["split_membership"] == "training"]
    statistics = compute_training_normalization_statistics(training_records, config=config)
    normalization_entry = write_normalization_statistics_atomic(
        output_run_dir / _NORMALIZATION_RELATIVE_PATH,
        statistics,
    )
    manifest = build_descriptor_artifacts_manifest(
        config=config,
        records=records,
        statistics=statistics,
        sample_entries=entries,
        normalization_entry=normalization_entry,
    )
    manifest["source_teacher_cache_manifest_file_sha256"] = (
        validated.source_teacher_cache_manifest_file_sha256
    )
    write_final_manifest_with_receipt_atomic(output_run_dir, manifest)
    verify_descriptor_artifact_collection(config=config, run_dir=output_run_dir)
    return DescriptorCollectionResult(
        run_dir=output_run_dir,
        manifest=MappingProxyType(dict(manifest)),
        source_teacher_cache_manifest_file_sha256=validated.source_teacher_cache_manifest_file_sha256,
        teacher_forward_count=0,
    )


def verify_descriptor_artifact_collection(
    *,
    config: DescriptorArtifactsConfig,
    run_dir: Path,
) -> VerifiedDescriptorCollection:
    """Verify a materialized descriptor collection from disk only."""

    resolved_run_dir = Path(run_dir).resolve()
    verify_final_manifest_receipt(resolved_run_dir)
    manifest_path = resolve_run_relative_artifact(
        run_dir=resolved_run_dir,
        relative_path=_FINAL_MANIFEST_NAME,
        expected_kind="final manifest",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "passed":
        _fail("B2_DESC_PARTIAL_CLAIMING_PASSED", "final manifest status must be passed")
    if manifest.get("teacher_forward_count") != 0:
        _fail("B2_DESC_TEACHER_FORWARD_COUNT_MISMATCH", "teacher_forward_count must equal zero")
    if (
        manifest.get("descriptor_contract_version") != config.descriptor_contract_version
        or manifest.get("normalization_contract_version")
        != config.normalization_contract_version
    ):
        _fail("B2_DESC_MANIFEST_INVALID", "manifest contract versions drifted")
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        _fail("B2_DESC_MANIFEST_INVALID", "passed manifest requires samples list")
    if len(samples) != sum(config.required_split_counts.values()):
        _fail("B2_DESC_CACHE_RECORD_COUNT_MISMATCH", "passed manifest must list 32 samples")
    expected_files = {
        _FINAL_MANIFEST_NAME,
        _FINAL_MANIFEST_RECEIPT_NAME,
        _NORMALIZATION_RELATIVE_PATH,
    }
    descriptor_records_by_id: dict[str, Mapping[str, Any]] = {}
    seen_relative_paths: set[str] = set()
    seen_targets: set[str] = set()
    for raw in samples:
        entry = PersistedDescriptorEntry(
            stable_sample_id=str(raw["stable_sample_id"]),
            relative_record_path=str(raw["relative_record_path"]),
            descriptor_record_scientific_sha256=str(raw["descriptor_record_scientific_sha256"]),
            descriptor_record_file_sha256=str(raw["descriptor_record_file_sha256"]),
            verification_status=str(raw.get("verification_status", "verified")),
        )
        if entry.verification_status != "verified":
            _fail("B2_DESC_PASSED_MANIFEST_REQUIRES_VERIFIED_DISK", "descriptor entry is not verified")
        normalized_relative = _normalized_relative_path(
            entry.relative_record_path,
            code="B2_DESC_RUN_RELATIVE_PATH_INVALID",
        )
        if normalized_relative in seen_relative_paths:
            _fail(
                "B2_DESC_RUN_RELATIVE_PATH_INVALID",
                f"duplicate manifest path entry: {normalized_relative}",
            )
        seen_relative_paths.add(normalized_relative)
        expected_files.add(normalized_relative)
        resolved_target = _run_relative_path_from_resolved(
            resolved_run_dir,
            resolve_run_relative_artifact(
                run_dir=resolved_run_dir,
                relative_path=normalized_relative,
                expected_kind=f"descriptor record {entry.stable_sample_id}",
            ),
        )
        if resolved_target in seen_targets:
            _fail(
                "B2_DESC_RUN_RELATIVE_PATH_INVALID",
                f"two manifest entries resolve to one artifact: {resolved_target}",
            )
        seen_targets.add(resolved_target)
        descriptor_records_by_id[entry.stable_sample_id] = verify_persisted_descriptor_entry(
            run_dir=resolved_run_dir,
            entry=entry,
            config=config,
        )
    normalization_relative = _normalized_relative_path(
        str(manifest["normalization_statistics_relative_path"]),
        code="B2_DESC_RUN_RELATIVE_PATH_INVALID",
    )
    if normalization_relative in seen_relative_paths:
        _fail(
            "B2_DESC_RUN_RELATIVE_PATH_INVALID",
            f"duplicate manifest path entry: {normalization_relative}",
        )
    expected_files.add(normalization_relative)
    expected_dirs = {
        str(PurePosixPath(path).parent)
        for path in expected_files
        if str(PurePosixPath(path).parent) != "."
    }
    actual_files: set[str] = set()
    actual_dirs: set[str] = set()
    for current_root, dir_names, file_names in os.walk(
        resolved_run_dir, topdown=True, followlinks=False
    ):
        current_path = Path(current_root)
        if current_path != resolved_run_dir:
            actual_dirs.add(current_path.relative_to(resolved_run_dir).as_posix())
        for name in dir_names:
            actual_dirs.add((current_path / name).relative_to(resolved_run_dir).as_posix())
        dir_names[:] = [
            name for name in dir_names if not (current_path / name).is_symlink()
        ]
        for file_name in file_names:
            actual_files.add(
                (current_path / file_name).relative_to(resolved_run_dir).as_posix()
            )
    if actual_files != expected_files or actual_dirs != expected_dirs:
        _fail("B2_DESC_ORPHAN_ARTIFACT", "run directory contains orphan, extra, or missing files")
    normalization_entry = PersistedNormalizationEntry(
        relative_path=str(manifest["normalization_statistics_relative_path"]),
        normalization_statistics_scientific_sha256=str(
            manifest["normalization_statistics_scientific_sha256"]
        ),
        normalization_statistics_file_sha256=str(
            manifest["normalization_statistics_file_sha256"]
        ),
    )
    normalization_statistics = verify_persisted_normalization_entry(
        run_dir=resolved_run_dir,
        entry=normalization_entry,
        config=config,
    )
    ordered_records = [
        descriptor_records_by_id[str(stable_id)] for stable_id in manifest["planned_stable_sample_ids"]
    ]
    _collection_provenance_from_records(ordered_records, config=config)
    if (
        manifest.get("expected_teacher_cache_scientific_sha256")
        != config.expected_teacher_cache_scientific_sha256
        or manifest.get("expected_split_scientific_sha256")
        != config.expected_split_scientific_sha256
        or manifest.get("expected_checkpoint_sha256") != config.expected_checkpoint_sha256
        or manifest.get("expected_execution_profile_sha256")
        != config.expected_execution_profile_sha256
        or not _is_sha256(manifest.get("source_teacher_cache_manifest_file_sha256", ""))
    ):
        _fail("B2_DESC_COLLECTION_PROVENANCE_MISMATCH", "manifest provenance pins drifted")
    if descriptor_sample_coverage_sha256(ordered_records) != manifest["descriptor_sample_coverage_sha256"]:
        _fail("B2_DESC_COLLECTION_HASH_MISMATCH", "descriptor sample coverage hash mismatch")
    if (
        descriptor_collection_scientific_sha256(
            records=ordered_records,
            statistics=normalization_statistics,
            config=config,
        )
        != manifest["descriptor_collection_scientific_sha256"]
    ):
        _fail("B2_DESC_COLLECTION_HASH_MISMATCH", "descriptor collection hash mismatch")
    if normalization_statistics["training_sample_coverage_sha256"] != manifest["training_sample_coverage_sha256"]:
        _fail("B2_DESC_COLLECTION_HASH_MISMATCH", "training coverage hash mismatch")
    if (
        normalization_statistics["normalization_training_coverage_sha256"]
        != manifest["normalization_training_coverage_sha256"]
    ):
        _fail("B2_DESC_COLLECTION_HASH_MISMATCH", "normalization training coverage mismatch")
    if (
        normalization_statistics["normalization_statistics_scientific_sha256"]
        != manifest["normalization_statistics_scientific_sha256"]
    ):
        _fail("B2_DESC_COLLECTION_HASH_MISMATCH", "normalization scientific hash mismatch")
    return VerifiedDescriptorCollection(
        run_dir=resolved_run_dir,
        manifest=MappingProxyType(dict(manifest)),
        descriptor_records_by_id=MappingProxyType(descriptor_records_by_id),
        normalization_statistics=MappingProxyType(dict(normalization_statistics)),
        teacher_forward_count=0,
    )


def compare_descriptor_artifact_collections(
    *,
    first: VerifiedDescriptorCollection,
    second: VerifiedDescriptorCollection,
) -> DescriptorCollectionComparison:
    """Compare two independently verified descriptor collections."""

    reasons: list[str] = []
    keys = (
        "descriptor_collection_scientific_sha256",
        "descriptor_sample_coverage_sha256",
        "normalization_statistics_scientific_sha256",
        "normalization_training_coverage_sha256",
    )
    for key in keys:
        if first.manifest[key] != second.manifest[key]:
            reasons.append(key)
    first_ids = sorted(first.descriptor_records_by_id)
    second_ids = sorted(second.descriptor_records_by_id)
    if first_ids != second_ids:
        reasons.append("stable_sample_ids")
    else:
        for stable_id in first_ids:
            first_record = first.descriptor_records_by_id[stable_id]
            second_record = second.descriptor_records_by_id[stable_id]
            if (
                first_record["descriptor_record_scientific_sha256"]
                != second_record["descriptor_record_scientific_sha256"]
            ):
                reasons.append(f"descriptor_record:{stable_id}")
                continue
            for depth in first_record["prediction_depths"]:
                if not torch.equal(
                    first_record["descriptor_by_depth"][depth],
                    second_record["descriptor_by_depth"][depth],
                ):
                    reasons.append(f"descriptor_tensor:{stable_id}:{depth}")
                    break
    if normalization_statistics_scientific_content(first.normalization_statistics) != normalization_statistics_scientific_content(second.normalization_statistics):
        reasons.append("normalization_statistics_values")
    first_files = {
        path.relative_to(first.run_dir).as_posix(): _sha256_file(path)
        for path in sorted(first.run_dir.rglob("*"))
        if path.is_file()
    }
    second_files = {
        path.relative_to(second.run_dir).as_posix(): _sha256_file(path)
        for path in sorted(second.run_dir.rglob("*"))
        if path.is_file()
    }
    file_byte_equal = first_files == second_files
    return DescriptorCollectionComparison(
        scientifically_equivalent=not reasons,
        reasons=tuple(reasons),
        file_byte_equal=file_byte_equal,
    )

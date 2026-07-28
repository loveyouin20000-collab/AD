"""Deterministic, source-only MVTec tiny-split construction."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, NoReturn

from rad.data.adapters import registry
from rad.data.adapters.mvtec import MVTecAdapter
from rad.runtime.execution_profile import (
    is_controlled_execution_profile_attestation,
)

_SPECIFICATION_PATH = "configs/phase_b/b2_tiny_gate_c.json"
_SPECIFICATION_SHA256 = (
    "06ceb68c7b89a70ce9ead5e38680c9fe158747dace337d3f901d48393bb7b630"
)
_PROFILE_SHA256 = "7af8dba39633743da0380fef9710940cded655f68c9efa8f84f5a52aeddb3c8d"
_BASE_COMMIT = "3a751b2784a50eb0a08ed49e1db2df0b53608ccc"
_BASE_TAG = "b1-strict-independent-v1"
_LEGACY_CANONICAL_HASH_V1 = (
    "0b9371deb6c55f359a14959c8b46ff50205191b1189a48ee380eafaf28c5791a"
)
_CANONICAL_SCIENTIFIC_HASH_V2 = (
    "91570da1fed6d7859d407196b10403581832ae0ff677a1ea7657ca76b91471f0"
)
_REJECTED_INTERMEDIATE_HASH = (
    "f840fd54f4385acda5af76f17d39e35251384f9ed56164b6b0769a0120ef6d88"
)
_HASH_MIGRATION = (
    "V1 mixed runtime provenance with science; f840fd54 was rejected because "
    "it retained branch/worktree fields; V2 uses a strict scientific whitelist."
)
_CATEGORIES = ("bottle", "carpet")
_SPLIT_ORDER = ("training", "calibration", "evaluation")
_FORBIDDEN_PATH_COMPONENTS = frozenset({"tests", "fixtures", "examples", "synthetic"})


class B2TinySplitError(RuntimeError):
    """A fail-closed B2 tiny-split contract error with a stable code."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2TinySplitError(code, detail)


@dataclass(frozen=True)
class SourceRecordsSnapshot:
    """Filesystem-validated source identities captured before pure construction."""

    adapter_module: str
    adapter_class: str
    adapter_request_name: str
    adapter_request_root: str
    adapter_request_count: int
    source_dataset: str
    source_split: str
    forbidden_target_dataset: str
    forbidden_target_access_count: int
    resolved_root: str
    canonical_records: tuple[Mapping[str, Any], ...]
    source_list_sha256: str
    dataset_root_identity_sha256: str
    specification_sha256: str


@dataclass(frozen=True)
class BuiltSplitManifest:
    """An official manifest and its canonical scientific digest."""

    manifest: Mapping[str, Any]
    scientific_sha256: str


def _thaw_for_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_for_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_for_json(item) for item in value]
    if isinstance(value, list):
        return [_thaw_for_json(item) for item in value]
    return value


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _thaw_for_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _fixed_specification() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "specification_id": "b2_tiny_gate_c",
        "transfer_direction": "mvtec_to_visa",
        "source_dataset": "mvtec",
        "forbidden_target_dataset": "visa",
        "categories": ["bottle", "carpet"],
        "splits": {
            "training": {
                "total": 16,
                "per_category": 8,
                "normal_per_category": 4,
                "anomalous_per_category": 4,
            },
            "calibration": {
                "total": 8,
                "per_category": 4,
                "normal_per_category": 2,
                "anomalous_per_category": 2,
            },
            "evaluation": {
                "total": 8,
                "per_category": 4,
                "normal_per_category": 2,
                "anomalous_per_category": 2,
            },
        },
        "total_selected_samples": 32,
        "label_requirements": {
            "normal": 0,
            "anomalous": 1,
            "allowed_labels": [0, 1],
            "stratify_by": ["category", "image_label"],
        },
        "mask_requirements": {
            "normal_mask_identity": None,
            "anomalous_mask_required": True,
            "anomalous_mask_must_exist": True,
        },
        "seed": 111,
        "stable_sample_id_rule": {
            "algorithm": "sha256",
            "canonical_encoding": "utf-8 compact sorted-key json",
            "identity_fields": [
                "dataset",
                "category",
                "source_split",
                "anomaly_type",
                "image_identity",
            ],
            "image_identity": "dataset-root-relative POSIX path",
            "ordering": "ascending stable_sample_id before seeded stratification",
        },
        "execution_profile": {
            "name": "frozen_deterministic_math",
            "path": "configs/execution/frozen_deterministic_math.json",
            "sha256": _PROFILE_SHA256,
        },
        "b1_base": {"tag": _BASE_TAG, "commit": _BASE_COMMIT},
        "fail_closed_requirements": [
            "production_registry_mvtec_adapter_only",
            "selected_categories_only",
            "unknown_category_rejected",
            "insufficient_stratum_rejected",
            "invalid_label_rejected",
            "missing_anomalous_mask_rejected",
            "stable_id_collision_rejected",
            "filesystem_order_independent",
            "pairwise_disjoint_memberships",
            "tests_fixtures_examples_synthetic_paths_rejected",
            "forbidden_target_not_enumerated_opened_or_hashed",
            "source_state_change_rejected",
            "execution_profile_identity_required",
            "immutable_runtime_attestation_required",
            "seed_drift_rejected",
            "output_collision_rejected",
            "atomic_official_write_required",
            "direct_tool_invocation_rejected",
        ],
    }


def _hash_contract() -> dict[str, Any]:
    return {
        "active_version": 2,
        "legacy_canonical_hash_v1": _LEGACY_CANONICAL_HASH_V1,
        "rejected_intermediate_candidate": _REJECTED_INTERMEDIATE_HASH,
        "canonical_scientific_hash_v2": _CANONICAL_SCIENTIFIC_HASH_V2,
        "migration": _HASH_MIGRATION,
    }


def _scientific_specification(specification: Mapping[str, Any]) -> dict[str, Any]:
    scientific = copy.deepcopy(dict(specification))
    scientific.pop("scientific_hash_contract", None)
    return scientific


def _validate_specification(specification: Mapping[str, Any]) -> None:
    categories = specification.get("categories")
    if categories != list(_CATEGORIES):
        _fail("B2_UNKNOWN_CATEGORY", f"categories must be {list(_CATEGORIES)}")
    profile = specification.get("execution_profile")
    if not isinstance(profile, Mapping) or profile.get("sha256") != _PROFILE_SHA256:
        _fail("B2_EXECUTION_PROFILE_MISMATCH", "execution profile hash is not approved")
    if specification.get("source_dataset") != "mvtec":
        _fail("B2_TARGET_DATASET_FORBIDDEN", "the only permitted source is MVTec")
    if (
        _scientific_specification(specification) != _fixed_specification()
        or specification.get("scientific_hash_contract") != _hash_contract()
    ):
        _fail("B2_SPECIFICATION_MISMATCH", "specification differs from the fixed contract")


def _relative_identity(path: Path, root: Path, *, code: str) -> str:
    try:
        identity = path.relative_to(root).as_posix()
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        _fail(code, f"path is missing or outside the dataset root: {path} ({exc})")
    if not identity or Path(identity).is_absolute():
        _fail(code, f"path does not have a relative identity: {path}")
    return identity


def _reject_forbidden_path(path: Path) -> None:
    if _FORBIDDEN_PATH_COMPONENTS.intersection(part.lower() for part in path.parts):
        _fail("B2_FORBIDDEN_SOURCE_PATH", f"forbidden source path: {path}")


def collect_source_records(
    *,
    source_root: Path | str,
    specification: Mapping[str, Any],
) -> SourceRecordsSnapshot:
    """Enumerate and validate MVTec identities without opening image contents."""

    _validate_specification(specification)
    root = Path(source_root)
    _reject_forbidden_path(root)
    adapter = registry.get_adapter("mvtec", root)
    if not isinstance(adapter, MVTecAdapter):
        _fail(
            "B2_PRODUCTION_ADAPTER_REQUIRED",
            "registry must return a production MVTecAdapter instance",
        )
    records = adapter.records("test", categories=_CATEGORIES)
    identity_fields = specification["stable_sample_id_rule"]["identity_fields"]
    canonical: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for record in records:
        if record.category not in _CATEGORIES:
            continue
        if record.dataset != "mvtec":
            _fail("B2_NON_SOURCE_RECORD", f"record dataset is {record.dataset!r}")
        if record.image_label not in (0, 1):
            _fail("B2_INVALID_LABEL", f"invalid image label {record.image_label!r}")
        if record.split != "test":
            _fail("B2_INVALID_SOURCE_SPLIT", f"invalid source split {record.split!r}")

        _reject_forbidden_path(record.image_path)
        image_identity = _relative_identity(
            record.image_path, root, code="B2_SOURCE_IMAGE_INVALID"
        )
        anomaly_type = record.image_path.parent.name
        mask_identity: str | None = None
        if record.image_label == 1:
            if record.mask_path is None:
                _fail("B2_ANOMALOUS_MASK_MISSING", f"mask absent for {image_identity}")
            _reject_forbidden_path(record.mask_path)
            mask_identity = _relative_identity(
                record.mask_path, root, code="B2_ANOMALOUS_MASK_INVALID"
            )
        elif record.mask_path is not None:
            _fail("B2_NORMAL_MASK_INVALID", f"normal record has mask: {image_identity}")

        available_identity = {
            "dataset": record.dataset,
            "category": record.category,
            "source_split": record.split,
            "anomaly_type": anomaly_type,
            "image_identity": image_identity,
        }
        stable_identity = {field: available_identity[field] for field in identity_fields}
        stable_sample_id = _canonical_sha256(stable_identity)
        if stable_sample_id in seen_ids:
            _fail("B2_STABLE_ID_COLLISION", f"duplicate stable ID {stable_sample_id}")
        seen_ids.add(stable_sample_id)
        canonical.append(
            {
                **available_identity,
                "image_label": record.image_label,
                "mask_identity": mask_identity,
                "stable_sample_id": stable_sample_id,
            }
        )

    canonical.sort(key=lambda item: item["stable_sample_id"])
    scientific_specification_bytes = (
        json.dumps(
            _scientific_specification(specification),
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    specification_sha256 = hashlib.sha256(scientific_specification_bytes).hexdigest()
    if specification_sha256 != _SPECIFICATION_SHA256:
        _fail(
            "B2_SPECIFICATION_HASH_MISMATCH",
            "tracked tiny-split specification bytes are not approved",
        )
    resolved_root = root.resolve(strict=True).as_posix()
    root_identity = {"dataset": "mvtec", "resolved_root": resolved_root}
    return SourceRecordsSnapshot(
        adapter_module="rad.data.adapters.mvtec",
        adapter_class="MVTecAdapter",
        adapter_request_name="mvtec",
        adapter_request_root=resolved_root,
        adapter_request_count=1,
        source_dataset="mvtec",
        source_split="test",
        forbidden_target_dataset="visa",
        forbidden_target_access_count=0,
        resolved_root=resolved_root,
        canonical_records=tuple(_deep_freeze(record) for record in canonical),
        source_list_sha256=_canonical_sha256(canonical),
        dataset_root_identity_sha256=_canonical_sha256(root_identity),
        specification_sha256=specification_sha256,
    )


def _validate_snapshot(snapshot: SourceRecordsSnapshot) -> None:
    if (
        snapshot.adapter_module != "rad.data.adapters.mvtec"
        or snapshot.adapter_class != "MVTecAdapter"
        or snapshot.adapter_request_name != "mvtec"
        or snapshot.adapter_request_root != snapshot.resolved_root
        or snapshot.adapter_request_count != 1
        or snapshot.source_dataset != "mvtec"
        or snapshot.source_split != "test"
        or snapshot.forbidden_target_dataset != "visa"
        or snapshot.forbidden_target_access_count != 0
        or snapshot.specification_sha256 != _SPECIFICATION_SHA256
    ):
        _fail("B2_SOURCE_SNAPSHOT_INVALID", "snapshot provenance is not approved")
    if not snapshot.resolved_root or not Path(snapshot.resolved_root).is_absolute():
        _fail("B2_SOURCE_SNAPSHOT_INVALID", "resolved dataset root is not absolute")
    expected_root_identity = _canonical_sha256(
        {"dataset": "mvtec", "resolved_root": snapshot.resolved_root}
    )
    if snapshot.dataset_root_identity_sha256 != expected_root_identity:
        _fail("B2_SOURCE_SNAPSHOT_INVALID", "dataset-root identity hash is invalid")

    records = list(snapshot.canonical_records)
    expected_fields = {
        "dataset",
        "category",
        "source_split",
        "anomaly_type",
        "image_identity",
        "image_label",
        "mask_identity",
        "stable_sample_id",
    }
    if any(
        not isinstance(record, Mapping)
        or isinstance(record, dict)
        or set(record) != expected_fields
        for record in records
    ):
        _fail("B2_SOURCE_SNAPSHOT_INVALID", "record schema or mutability is invalid")
    if records != sorted(records, key=lambda item: item["stable_sample_id"]):
        _fail("B2_SOURCE_SNAPSHOT_INVALID", "source records are not canonically ordered")
    if _canonical_sha256(records) != snapshot.source_list_sha256:
        _fail("B2_SOURCE_SNAPSHOT_INVALID", "source-list hash does not match records")
    seen: set[str] = set()
    identity_fields = _fixed_specification()["stable_sample_id_rule"]["identity_fields"]
    for record in records:
        image_identity = record["image_identity"]
        mask_identity = record["mask_identity"]
        label = record["image_label"]
        if (
            record["dataset"] != "mvtec"
            or record["category"] not in _CATEGORIES
            or record["source_split"] != "test"
            or label not in (0, 1)
            or not isinstance(image_identity, str)
            or not image_identity
            or PurePosixPath(image_identity).is_absolute()
            or record["anomaly_type"] != PurePosixPath(image_identity).parent.name
            or _FORBIDDEN_PATH_COMPONENTS.intersection(
                component.lower() for component in PurePosixPath(image_identity).parts
            )
        ):
            _fail("B2_SOURCE_SNAPSHOT_INVALID", "record source identity is invalid")
        if label == 0 and mask_identity is not None:
            _fail("B2_SOURCE_SNAPSHOT_INVALID", "normal record has a mask identity")
        if label == 1 and (
            not isinstance(mask_identity, str)
            or not mask_identity
            or PurePosixPath(mask_identity).is_absolute()
            or _FORBIDDEN_PATH_COMPONENTS.intersection(
                component.lower() for component in PurePosixPath(mask_identity).parts
            )
        ):
            _fail("B2_SOURCE_SNAPSHOT_INVALID", "anomalous mask identity is invalid")
        stable_identity = {field: record[field] for field in identity_fields}
        stable_id = _canonical_sha256(stable_identity)
        if stable_id != record["stable_sample_id"] or stable_id in seen:
            _fail("B2_SOURCE_SNAPSHOT_INVALID", "record identity is invalid or duplicated")
        seen.add(stable_id)


def _attestation_provenance(attestation: Any) -> dict[str, str]:
    if not is_controlled_execution_profile_attestation(attestation):
        _fail("B2_RUNTIME_ATTESTATION_REQUIRED", "controlled runtime attestation required")
    canonical = attestation.canonical_attestation()
    recomputed_sha256 = _canonical_sha256(canonical)
    if recomputed_sha256 != attestation.attestation_sha256:
        _fail(
            "B2_RUNTIME_ATTESTATION_INVALID",
            "runtime attestation digest does not match canonical evidence",
        )
    profile = canonical.get("profile")
    canary = canonical.get("canary")
    if (
        not isinstance(profile, Mapping)
        or profile.get("profile_id") != "frozen_deterministic_math"
        or profile.get("expected_sha256") != _PROFILE_SHA256
        or profile.get("launcher_sha256") != _PROFILE_SHA256
        or profile.get("runtime_sha256") != _PROFILE_SHA256
        or profile.get("hashes_match") is not True
        or not isinstance(canary, Mapping)
        or canary.get("self_repeatability") is not True
        or canary.get("independent_reconstruction") is not True
    ):
        _fail("B2_EXECUTION_PROFILE_MISMATCH", "runtime provenance is not exact")
    provenance = dict(attestation.artifact_provenance())
    expected = {
        "execution_profile_name": "frozen_deterministic_math",
        "execution_profile_sha256": _PROFILE_SHA256,
        "runtime_attestation_sha256": attestation.attestation_sha256,
    }
    if provenance != expected:
        _fail("B2_EXECUTION_PROFILE_MISMATCH", "artifact provenance is not exact")
    return provenance


def _validate_repository_identity(identity: Mapping[str, Any]) -> None:
    generation_commit = identity.get("generation_git_commit")
    if (
        identity.get("b1_base_tag") != _BASE_TAG
        or identity.get("b1_base_commit") != _BASE_COMMIT
        or not isinstance(generation_commit, str)
        or len(generation_commit) != 40
        or any(character not in "0123456789abcdef" for character in generation_commit)
        or not isinstance(identity.get("generation_branch"), str)
        or not isinstance(identity.get("worktree_path"), str)
        or not isinstance(identity.get("worktree_clean"), bool)
    ):
        _fail("B2_REPOSITORY_IDENTITY_MISMATCH", "repository identity drifted")


def _selected_splits(
    records: Sequence[Mapping[str, Any]], specification: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(specification["seed"])
    selected: dict[str, list[dict[str, Any]]] = {
        split: [] for split in _SPLIT_ORDER
    }
    for category in _CATEGORIES:
        for label in (0, 1):
            stratum = [
                record
                for record in records
                if record["category"] == category and record["image_label"] == label
            ]
            stratum.sort(key=lambda item: item["stable_sample_id"])
            rng.shuffle(stratum)
            needed = sum(
                specification["splits"][split][
                    "normal_per_category" if label == 0 else "anomalous_per_category"
                ]
                for split in _SPLIT_ORDER
            )
            if len(stratum) < needed:
                _fail(
                    "B2_INSUFFICIENT_STRATUM",
                    f"{category}/label={label}: need {needed}, found {len(stratum)}",
                )
            offset = 0
            for split in _SPLIT_ORDER:
                count = specification["splits"][split][
                    "normal_per_category" if label == 0 else "anomalous_per_category"
                ]
                for source in stratum[offset : offset + count]:
                    selected[split].append(
                        {
                            "stable_sample_id": source["stable_sample_id"],
                            "category": source["category"],
                            "image_label": source["image_label"],
                            "anomaly_type": source["anomaly_type"],
                            "image_identity": source["image_identity"],
                            "mask_identity": source["mask_identity"],
                            "membership": split,
                        }
                    )
                offset += count
    return selected


def build_split_manifest(
    *,
    specification: Mapping[str, Any],
    source_snapshot: SourceRecordsSnapshot,
    runtime_attestation: Any,
    repository_identity: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
) -> BuiltSplitManifest:
    """Purely construct and audit the fixed B2 tiny-split manifest."""

    _validate_specification(specification)
    _validate_snapshot(source_snapshot)
    _validate_repository_identity(repository_identity)
    provenance = _attestation_provenance(runtime_attestation)
    selected = _selected_splits(source_snapshot.canonical_records, specification)
    source_only_audit = {
        "passed": (
            source_snapshot.adapter_request_name == "mvtec"
            and source_snapshot.adapter_request_count == 1
            and source_snapshot.source_dataset == "mvtec"
            and source_snapshot.source_split == "test"
            and source_snapshot.forbidden_target_dataset == "visa"
            and source_snapshot.forbidden_target_access_count == 0
            and all(
                record["dataset"] == source_snapshot.source_dataset
                for record in source_snapshot.canonical_records
            )
        ),
        "source_dataset": source_snapshot.source_dataset,
        "forbidden_target_dataset": source_snapshot.forbidden_target_dataset,
        "forbidden_target_access_count": source_snapshot.forbidden_target_access_count,
    }
    split_totals = {split: len(selected[split]) for split in _SPLIT_ORDER}
    total_selected = sum(split_totals.values())
    count_by_split: dict[str, dict[str, dict[str, int]]] = {}
    count_invariants_passed = total_selected == specification["total_selected_samples"]
    for split in _SPLIT_ORDER:
        count_by_split[split] = {}
        for category in _CATEGORIES:
            category_rows = [
                sample for sample in selected[split] if sample["category"] == category
            ]
            normal = sum(sample["image_label"] == 0 for sample in category_rows)
            anomalous = sum(sample["image_label"] == 1 for sample in category_rows)
            counts = {
                "total": len(category_rows),
                "normal": normal,
                "anomalous": anomalous,
            }
            count_by_split[split][category] = counts
            expected_split = specification["splits"][split]
            count_invariants_passed = count_invariants_passed and counts == {
                "total": expected_split["per_category"],
                "normal": expected_split["normal_per_category"],
                "anomalous": expected_split["anomalous_per_category"],
            }
        count_invariants_passed = (
            count_invariants_passed
            and split_totals[split] == specification["splits"][split]["total"]
            and all(sample["membership"] == split for sample in selected[split])
        )
    count_audit = {
        "passed": count_invariants_passed,
        "total_selected": total_selected,
        "split_totals": split_totals,
        "by_split": count_by_split,
    }

    ids_by_split = {
        split: {sample["stable_sample_id"] for sample in selected[split]}
        for split in _SPLIT_ORDER
    }
    intersections = {
        "training_calibration": sorted(
            ids_by_split["training"] & ids_by_split["calibration"]
        ),
        "training_evaluation": sorted(
            ids_by_split["training"] & ids_by_split["evaluation"]
        ),
        "calibration_evaluation": sorted(
            ids_by_split["calibration"] & ids_by_split["evaluation"]
        ),
    }
    all_ids = [
        sample["stable_sample_id"]
        for split in _SPLIT_ORDER
        for sample in selected[split]
    ]
    unique_selected_ids = len(set(all_ids))
    pairwise_disjoint = all(not overlap for overlap in intersections.values())
    overlap_audit = {
        "passed": pairwise_disjoint and unique_selected_ids == total_selected,
        "pairwise_disjoint": pairwise_disjoint,
        "unique_selected_ids": unique_selected_ids,
        "intersections": intersections,
    }

    forbidden_components = sorted(_FORBIDDEN_PATH_COMPONENTS)
    fixture_violations = sorted(
        {
            identity
            for split in _SPLIT_ORDER
            for sample in selected[split]
            for identity in (sample["image_identity"], sample["mask_identity"])
            if identity is not None
            and _FORBIDDEN_PATH_COMPONENTS.intersection(
                component.lower() for component in identity.split("/")
            )
        }
    )
    fixture_path_audit = {
        "passed": not fixture_violations,
        "forbidden_components": forbidden_components,
        "violations": fixture_violations,
    }

    selected_rows = [
        sample for split in _SPLIT_ORDER for sample in selected[split]
    ]
    anomalous_rows = [
        sample for sample in selected_rows if sample["image_label"] == 1
    ]
    normal_rows = [sample for sample in selected_rows if sample["image_label"] == 0]
    anomalous_with_masks = sum(
        bool(sample["mask_identity"]) for sample in anomalous_rows
    )
    normal_with_masks = sum(bool(sample["mask_identity"]) for sample in normal_rows)
    mask_audit = {
        "passed": (
            anomalous_with_masks == len(anomalous_rows) and normal_with_masks == 0
        ),
        "anomalous_selected": len(anomalous_rows),
        "anomalous_with_masks": anomalous_with_masks,
        "normal_selected": len(normal_rows),
        "normal_with_masks": normal_with_masks,
    }

    if not count_audit["passed"]:
        _fail("B2_SPLIT_COUNT_AUDIT_FAILED", "selected rows violate fixed quotas")
    if not overlap_audit["passed"]:
        _fail("B2_SPLIT_OVERLAP_AUDIT_FAILED", "selected memberships overlap")
    if not fixture_path_audit["passed"]:
        _fail("B2_FORBIDDEN_SOURCE_PATH", "selected rows contain forbidden paths")
    if not mask_audit["passed"]:
        _fail("B2_MASK_AUDIT_FAILED", "selected rows violate mask invariants")
    if not source_only_audit["passed"]:
        _fail("B2_SOURCE_ONLY_AUDIT_FAILED", "snapshot evidence is not source-only")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "run_id": run_metadata["run_id"],
        "creation_timestamp": run_metadata["creation_timestamp"],
        "output_directory": run_metadata["output_directory"],
        "b1_base_tag": repository_identity["b1_base_tag"],
        "b1_base_commit": repository_identity["b1_base_commit"],
        "generation_git_commit": repository_identity["generation_git_commit"],
        "generation_branch": repository_identity["generation_branch"],
        "generation_worktree_path": repository_identity["worktree_path"],
        "worktree_clean": repository_identity["worktree_clean"],
        "transfer_direction": "mvtec_to_visa",
        "source": {
            "dataset": "mvtec",
            "adapter_module": source_snapshot.adapter_module,
            "adapter_class": source_snapshot.adapter_class,
            "record_count": len(source_snapshot.canonical_records),
            "source_list_sha256": source_snapshot.source_list_sha256,
            "dataset_root_identity_sha256": source_snapshot.dataset_root_identity_sha256,
        },
        "forbidden_target_dataset": "visa",
        "categories": list(_CATEGORIES),
        "seed": 111,
        "specification": {
            "specification_id": "b2_tiny_gate_c",
            "path": _SPECIFICATION_PATH,
            "sha256": source_snapshot.specification_sha256,
        },
        "execution_profile": provenance,
        "runtime_attestation": _thaw_for_json(
            runtime_attestation.canonical_attestation()
        ),
        "splits": selected,
        "count_audit": count_audit,
        "overlap_audit": overlap_audit,
        "source_only_audit": source_only_audit,
        "fixture_path_audit": fixture_path_audit,
        "mask_audit": mask_audit,
    }
    scientific_sha256 = canonical_scientific_hash_v2(manifest)
    manifest["scientific_hash_contract"] = {
        "active_version": 2,
        "legacy_canonical_hash_v1": _LEGACY_CANONICAL_HASH_V1,
        "rejected_intermediate_candidate": _REJECTED_INTERMEDIATE_HASH,
        "canonical_scientific_hash_v2": scientific_sha256,
        "migration": _HASH_MIGRATION,
    }
    return BuiltSplitManifest(manifest=manifest, scientific_sha256=scientific_sha256)


def canonical_scientific_content_v2(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Project the strict V2 whitelist from current or historical manifests."""

    historical_base = manifest.get("base")
    if not isinstance(historical_base, Mapping):
        historical_base = {}
    source = manifest["source"]
    specification = manifest["specification"]
    execution_profile = manifest["execution_profile"]
    selected_samples = [
        {
            "stable_sample_id": row["stable_sample_id"],
            "category": row["category"],
            "image_label": row["image_label"],
            "mask_identity": row["mask_identity"],
            "membership": row["membership"],
        }
        for split in _SPLIT_ORDER
        for row in manifest["splits"][split]
    ]
    return {
        "b1_base_tag": manifest.get("b1_base_tag", historical_base.get("tag")),
        "b1_base_commit": manifest.get("b1_base_commit", historical_base.get("commit")),
        "transfer_direction": manifest["transfer_direction"],
        "source_dataset": source["dataset"],
        "source_categories": copy.deepcopy(manifest["categories"]),
        "seed": manifest["seed"],
        "split_specification_sha256": specification["sha256"],
        "enumerated_source_list_sha256": source["source_list_sha256"],
        "execution_profile_sha256": execution_profile["execution_profile_sha256"],
        "selected_samples": selected_samples,
    }


def canonical_scientific_hash_v2(manifest: Mapping[str, Any]) -> str:
    """Hash V2 canonical scientific content without machine provenance."""

    return _canonical_sha256(canonical_scientific_content_v2(manifest))


def canonical_scientific_content(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility alias for the active V2 scientific-content contract."""

    return canonical_scientific_content_v2(manifest)


def canonical_scientific_sha256(manifest: Mapping[str, Any]) -> str:
    """Compatibility alias for the active V2 scientific hash."""

    return canonical_scientific_hash_v2(manifest)

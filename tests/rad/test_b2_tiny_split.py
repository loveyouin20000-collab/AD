"""B2-01 RED: deterministic, source-only MVTec tiny-split contract.

The production module is imported inside each test so a missing implementation is
reported as an ordinary pytest failure, not as a collection error.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from rad.data.adapters.mvtec import MVTecAdapter
from rad.data.adapters.types import EvaluationRecord

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "configs" / "phase_b" / "b2_tiny_gate_c.json"
PROFILE_PATH = REPO_ROOT / "configs" / "execution" / "frozen_deterministic_math.json"
EXPECTED_PROFILE_SHA256 = (
    "7af8dba39633743da0380fef9710940cded655f68c9efa8f84f5a52aeddb3c8d"
)
EXPECTED_BASE_COMMIT = "3a751b2784a50eb0a08ed49e1db2df0b53608ccc"
EXPECTED_SPLIT_COUNTS = {"training": 16, "calibration": 8, "evaluation": 8}
EXPECTED_PER_CATEGORY = {"training": 8, "calibration": 4, "evaluation": 4}
EXPECTED_PER_CATEGORY_LABEL = {"training": 4, "calibration": 2, "evaluation": 2}
REQUIRED_API = {
    "B2TinySplitError",
    "BuiltSplitManifest",
    "build_split_manifest",
    "canonical_scientific_content",
    "canonical_scientific_sha256",
    "collect_source_records",
}


def _subject() -> ModuleType:
    try:
        module = importlib.import_module("rad.phase_b.b2_tiny_split")
    except ModuleNotFoundError as exc:
        if exc.name in {"rad.phase_b", "rad.phase_b.b2_tiny_split"}:
            pytest.fail(
                "B2-01 RED: missing production module rad.phase_b.b2_tiny_split",
                pytrace=False,
            )
        raise
    missing = sorted(name for name in REQUIRED_API if not hasattr(module, name))
    assert not missing, f"B2-01 API is incomplete: {missing}"
    return module


def _specification() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record(
    root: Path,
    *,
    category: str,
    label: int,
    index: int,
    path_prefix: tuple[str, ...] = (),
    mask: bool = True,
) -> EvaluationRecord:
    anomaly_type = "good" if label == 0 else "crack"
    image_path = (
        root
        .joinpath(*path_prefix)
        .joinpath(category, "test", anomaly_type, f"{index:03d}.png")
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"b2-controlled-image")
    mask_path = None
    if label == 1 and mask:
        mask_path = (
            root
            .joinpath(*path_prefix)
            .joinpath(
                category,
                "ground_truth",
                anomaly_type,
                f"{index:03d}_mask.png",
            )
        )
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.write_bytes(b"b2-controlled-mask")
    return EvaluationRecord(
        sample_id=image_path.relative_to(root).as_posix(),
        dataset="mvtec",
        category=category,
        image_path=image_path,
        mask_path=mask_path,
        image_label=label,
        split="test",
    )


def _complete_records(root: Path) -> tuple[EvaluationRecord, ...]:
    return tuple(
        _record(root, category=category, label=label, index=index)
        for category in ("bottle", "carpet")
        for label in (0, 1)
        for index in range(10)
    )


class _ControlledMVTecAdapter(MVTecAdapter):
    def __init__(self, records: Sequence[EvaluationRecord]) -> None:
        self._records = tuple(records)
        self.records_calls: list[str] = []
        self.open_image_calls = 0
        self.open_mask_calls = 0

    def records(
        self,
        split: str = "test",
        *,
        categories: Sequence[str] | None = None,
    ) -> Sequence[EvaluationRecord]:
        self.records_calls.append(split)
        if categories is None:
            return self._records
        selected = set(categories)
        return tuple(record for record in self._records if record.category in selected)

    def open_image(self, record: EvaluationRecord) -> None:
        del record
        self.open_image_calls += 1
        raise AssertionError("tiny split must not open source images")

    def open_mask(self, record: EvaluationRecord) -> None:
        del record
        self.open_mask_calls += 1
        raise AssertionError("tiny split must not open source masks")


class _NotProductionAdapter:
    def __init__(self, records: Sequence[EvaluationRecord]) -> None:
        self._records = tuple(records)

    def records(self, split: str = "test") -> Sequence[EvaluationRecord]:
        del split
        return self._records


class _ForbiddenVisAAdapter:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("VisA adapter construction is forbidden")

    def records(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("VisA sample enumeration is forbidden")

    def open_image(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("VisA image access is forbidden")

    def open_mask(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("VisA mask access is forbidden")


class _ControlledAttestationFixture:
    def __init__(self) -> None:
        self.attestation_sha256 = "a" * 64


def _controlled_attestation() -> _ControlledAttestationFixture:
    return _ControlledAttestationFixture()


def _install_attestation_boundary(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def controlled_provenance(attestation: Any) -> dict[str, str]:
        if not isinstance(attestation, _ControlledAttestationFixture):
            raise module.B2TinySplitError(
                "B2_RUNTIME_ATTESTATION_REQUIRED",
                "controlled runtime attestation required",
            )
        if attestation.attestation_sha256 != "a" * 64:
            raise module.B2TinySplitError(
                "B2_RUNTIME_ATTESTATION_INVALID",
                "runtime attestation hash mismatch",
            )
        return {
            "execution_profile_name": "frozen_deterministic_math",
            "execution_profile_sha256": EXPECTED_PROFILE_SHA256,
            "runtime_attestation_sha256": attestation.attestation_sha256,
        }

    monkeypatch.setattr(module, "_attestation_provenance", controlled_provenance)


def _repository_identity() -> dict[str, Any]:
    return {
        "base_tag": "b1-strict-independent-v1",
        "base_commit": EXPECTED_BASE_COMMIT,
        "worktree_path": "/root/autodl-tmp/AD-phase-b2-gate-c",
        "branch": "phase-b2-tiny-gate-c",
        "worktree_git_sha": EXPECTED_BASE_COMMIT,
    }


def _run_metadata(suffix: str = "a") -> dict[str, str]:
    second = "01" if suffix == "a" else "02"
    return {
        "run_id": f"b2-gate-c-{suffix}",
        "creation_timestamp": f"2026-07-21T00:00:{second}Z",
        "output_directory": f"artifacts/phase_b/b2_gate_c/{suffix}",
    }


def _install_registry_spy(
    monkeypatch: pytest.MonkeyPatch,
    adapter: _ControlledMVTecAdapter,
    calls: list[tuple[str, Path]],
) -> None:
    def get_adapter(name: str, root: Path | str) -> _ControlledMVTecAdapter:
        calls.append((name, Path(root)))
        if name != "mvtec":
            return _ForbiddenVisAAdapter()  # type: ignore[return-value]
        return adapter

    monkeypatch.setattr("rad.data.adapters.registry.get_adapter", get_adapter)
    monkeypatch.setitem(
        importlib.import_module("rad.data.adapters.registry")._ADAPTER_FACTORIES,
        "visa",
        _ForbiddenVisAAdapter,
    )


def _collect(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    records: Sequence[EvaluationRecord],
) -> tuple[Any, _ControlledMVTecAdapter, list[tuple[str, Path]]]:
    _install_attestation_boundary(module, monkeypatch)
    adapter = _ControlledMVTecAdapter(records)
    calls: list[tuple[str, Path]] = []
    _install_registry_spy(monkeypatch, adapter, calls)
    snapshot = module.collect_source_records(
        source_root=root,
        specification=_specification(),
    )
    return snapshot, adapter, calls


def _build(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    records: Sequence[EvaluationRecord] | None = None,
    *,
    specification: Mapping[str, Any] | None = None,
    runtime_attestation: Any | None = None,
    run_suffix: str = "a",
) -> Any:
    snapshot, _, _ = _collect(
        module,
        monkeypatch,
        root,
        _complete_records(root) if records is None else records,
    )
    return module.build_split_manifest(
        specification=_specification() if specification is None else specification,
        source_snapshot=snapshot,
        runtime_attestation=(
            _controlled_attestation()
            if runtime_attestation is None
            else runtime_attestation
        ),
        repository_identity=_repository_identity(),
        run_metadata=_run_metadata(run_suffix),
    )


def _manifest(result: Any) -> Mapping[str, Any]:
    manifest = result.manifest
    assert isinstance(manifest, Mapping)
    return manifest


def _samples_by_split(manifest: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    return {
        split: list(manifest["splits"][split])
        for split in ("training", "calibration", "evaluation")
    }


def _assert_b2_error(module: ModuleType, code: str, operation: Callable[[], Any]) -> None:
    with pytest.raises(module.B2TinySplitError, match=code):
        operation()


def test_expected_module_exposes_clear_builder_api() -> None:
    _subject()


def test_tracked_specification_pins_the_complete_tiny_split_contract() -> None:
    module = _subject()
    del module
    spec = _specification()
    assert spec["specification_id"] == "b2_tiny_gate_c"
    assert spec["transfer_direction"] == "mvtec_to_visa"
    assert spec["source_dataset"] == "mvtec"
    assert spec["forbidden_target_dataset"] == "visa"
    assert spec["categories"] == ["bottle", "carpet"]
    assert spec["splits"] == {
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
    }
    assert spec["total_selected_samples"] == 32
    assert spec["seed"] == 111
    assert spec["execution_profile"]["sha256"] == EXPECTED_PROFILE_SHA256
    assert spec["b1_base"] == {
        "tag": "b1-strict-independent-v1",
        "commit": EXPECTED_BASE_COMMIT,
    }


def test_exact_16_8_8_category_and_label_quotas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _subject()
    manifest = _manifest(_build(module, monkeypatch, tmp_path / "controlled_mvtec"))
    by_split = _samples_by_split(manifest)
    assert {name: len(samples) for name, samples in by_split.items()} == (
        EXPECTED_SPLIT_COUNTS
    )
    for split, samples in by_split.items():
        categories = Counter(sample["category"] for sample in samples)
        strata = Counter(
            (sample["category"], sample["image_label"]) for sample in samples
        )
        assert categories == {
            "bottle": EXPECTED_PER_CATEGORY[split],
            "carpet": EXPECTED_PER_CATEGORY[split],
        }
        assert strata == {
            ("bottle", 0): EXPECTED_PER_CATEGORY_LABEL[split],
            ("bottle", 1): EXPECTED_PER_CATEGORY_LABEL[split],
            ("carpet", 0): EXPECTED_PER_CATEGORY_LABEL[split],
            ("carpet", 1): EXPECTED_PER_CATEGORY_LABEL[split],
        }


def test_every_split_contains_bottle_and_carpet_and_memberships_are_disjoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _subject()
    manifest = _manifest(_build(module, monkeypatch, tmp_path / "controlled_mvtec"))
    by_split = _samples_by_split(manifest)
    memberships = {
        split: {sample["stable_sample_id"] for sample in samples}
        for split, samples in by_split.items()
    }
    assert all(
        {sample["category"] for sample in samples} == {"bottle", "carpet"}
        for samples in by_split.values()
    )
    assert memberships["training"].isdisjoint(memberships["calibration"])
    assert memberships["training"].isdisjoint(memberships["evaluation"])
    assert memberships["calibration"].isdisjoint(memberships["evaluation"])
    assert len(set().union(*memberships.values())) == 32


def test_anomalous_samples_have_mask_identity_and_normal_samples_do_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _subject()
    manifest = _manifest(_build(module, monkeypatch, tmp_path / "controlled_mvtec"))
    samples = [
        sample
        for split_samples in _samples_by_split(manifest).values()
        for sample in split_samples
    ]
    assert all(
        sample["mask_identity"] is None
        if sample["image_label"] == 0
        else isinstance(sample["mask_identity"], str) and sample["mask_identity"]
        for sample in samples
    )


def test_collection_uses_only_production_registry_mvtec_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _subject()
    root = tmp_path / "controlled_mvtec"
    _, adapter, calls = _collect(module, monkeypatch, root, _complete_records(root))
    assert calls == [("mvtec", root)]
    assert adapter.records_calls == ["test"]
    assert adapter.open_image_calls == 0
    assert adapter.open_mask_calls == 0


def test_collection_uses_real_production_mvtec_adapter(
    tmp_path: Path,
) -> None:
    module = _subject()
    root = tmp_path / "controlled_mvtec"
    _complete_records(root)
    snapshot = module.collect_source_records(
        source_root=root,
        specification=_specification(),
    )
    assert snapshot.adapter_module == "rad.data.adapters.mvtec"
    assert snapshot.adapter_class == "MVTecAdapter"
    assert len(snapshot.canonical_records) == 40
    assert all(record["dataset"] == "mvtec" for record in snapshot.canonical_records)


def test_collection_rejects_nonproduction_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _subject()
    root = tmp_path / "controlled_mvtec"
    records = _complete_records(root)
    monkeypatch.setattr(
        "rad.data.adapters.registry.get_adapter",
        lambda _name, _root: _NotProductionAdapter(records),
    )
    _assert_b2_error(
        module,
        "B2_PRODUCTION_ADAPTER_REQUIRED",
        lambda: module.collect_source_records(
            source_root=root,
            specification=_specification(),
        ),
    )


def test_source_snapshot_is_deeply_immutable_and_provenance_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _subject()
    root = tmp_path / "controlled_mvtec"
    snapshot, _, _ = _collect(module, monkeypatch, root, _complete_records(root))
    with pytest.raises(TypeError):
        snapshot.canonical_records[0]["category"] = "forged"
    for field, forged in (
        ("adapter_module", "forged.adapter"),
        ("adapter_class", "ForgedAdapter"),
        ("specification_sha256", "0" * 64),
        ("dataset_root_identity_sha256", "0" * 64),
    ):
        forged_snapshot = replace(snapshot, **{field: forged})
        _assert_b2_error(
            module,
            "B2_SOURCE_SNAPSHOT_INVALID",
            lambda forged_snapshot=forged_snapshot: module.build_split_manifest(
                specification=_specification(),
                source_snapshot=forged_snapshot,
                runtime_attestation=_controlled_attestation(),
                repository_identity=_repository_identity(),
                run_metadata=_run_metadata(),
            ),
        )


def test_visa_adapter_and_sample_operations_are_never_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _subject()
    root = tmp_path / "controlled_mvtec"
    result = _build(module, monkeypatch, root)
    assert _manifest(result)["source"]["dataset"] == "mvtec"


@pytest.mark.parametrize("forbidden_component", ["tests", "fixtures", "examples", "synthetic"])
def test_rejects_fixture_example_and_synthetic_record_paths(
    forbidden_component: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _subject()
    root = tmp_path / "controlled_mvtec"
    records = list(_complete_records(root))
    records[0] = _record(
        root,
        category="bottle",
        label=0,
        index=100,
        path_prefix=(forbidden_component,),
    )
    _assert_b2_error(
        module,
        "B2_FORBIDDEN_SOURCE_PATH",
        lambda: _build(module, monkeypatch, root, records),
    )


def test_unknown_category_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _subject()
    root = tmp_path / "controlled_mvtec"
    specification = _specification()
    specification["categories"] = ["bottle", "screw"]
    _assert_b2_error(
        module,
        "B2_UNKNOWN_CATEGORY",
        lambda: _build(
            module,
            monkeypatch,
            root,
            specification=specification,
        ),
    )


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        ("image_label", 2, "B2_INVALID_LABEL"),
        ("dataset", "visa", "B2_NON_SOURCE_RECORD"),
        ("split", "train", "B2_INVALID_SOURCE_SPLIT"),
    ],
)
def test_invalid_record_identity_fails_closed(
    field: str,
    value: Any,
    expected_code: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _subject()
    root = tmp_path / "controlled_mvtec"
    records = list(_complete_records(root))
    records[0] = replace(records[0], **{field: value})
    _assert_b2_error(
        module,
        expected_code,
        lambda: _build(module, monkeypatch, root, records),
    )


def test_insufficient_stratum_fails_instead_of_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _subject()
    root = tmp_path / "controlled_mvtec"
    records = list(_complete_records(root))
    records = [
        record
        for record in records
        if not (
            record.category == "bottle"
            and record.image_label == 1
            and record.image_path.stem in {"000", "001", "002"}
        )
    ]
    _assert_b2_error(
        module,
        "B2_INSUFFICIENT_STRATUM",
        lambda: _build(module, monkeypatch, root, records),
    )


def test_missing_anomalous_mask_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _subject()
    root = tmp_path / "controlled_mvtec"
    records = list(_complete_records(root))
    records[8] = _record(root, category="bottle", label=1, index=0, mask=False)
    _assert_b2_error(
        module,
        "B2_ANOMALOUS_MASK_MISSING",
        lambda: _build(module, monkeypatch, root, records),
    )


def test_nonexistent_anomalous_mask_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _subject()
    root = tmp_path / "controlled_mvtec"
    records = list(_complete_records(root))
    anomalous_index = next(
        index for index, record in enumerate(records) if record.image_label == 1
    )
    missing = root / "bottle" / "ground_truth" / "crack" / "missing_mask.png"
    records[anomalous_index] = replace(records[anomalous_index], mask_path=missing)
    _assert_b2_error(
        module,
        "B2_ANOMALOUS_MASK_INVALID",
        lambda: _build(module, monkeypatch, root, records),
    )


def test_execution_profile_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _subject()
    spec = _specification()
    spec["execution_profile"]["sha256"] = "0" * 64
    _assert_b2_error(
        module,
        "B2_EXECUTION_PROFILE_MISMATCH",
        lambda: _build(
            module,
            monkeypatch,
            tmp_path / "controlled_mvtec",
            specification=spec,
        ),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_tag", "wrong-tag"),
        ("base_commit", "0" * 40),
        ("branch", "wrong-branch"),
        ("worktree_path", "/tmp/wrong-worktree"),
    ],
)
def test_repository_identity_drift_fails_closed(
    field: str,
    value: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _subject()
    root = tmp_path / "controlled_mvtec"
    snapshot, _, _ = _collect(module, monkeypatch, root, _complete_records(root))
    identity = _repository_identity()
    identity[field] = value
    _assert_b2_error(
        module,
        "B2_REPOSITORY_IDENTITY_MISMATCH",
        lambda: module.build_split_manifest(
            specification=_specification(),
            source_snapshot=snapshot,
            runtime_attestation=_controlled_attestation(),
            repository_identity=identity,
            run_metadata=_run_metadata(),
        ),
    )


@pytest.mark.parametrize("forged", [None, {}, {"attestation_sha256": "0" * 64}])
def test_missing_or_forged_runtime_attestation_is_rejected(
    forged: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _subject()
    root = tmp_path / "controlled_mvtec"
    snapshot, _, _ = _collect(module, monkeypatch, root, _complete_records(root))
    _assert_b2_error(
        module,
        "B2_RUNTIME_ATTESTATION_REQUIRED",
        lambda: module.build_split_manifest(
            specification=_specification(),
            source_snapshot=snapshot,
            runtime_attestation=forged,
            repository_identity=_repository_identity(),
            run_metadata=_run_metadata(),
        ),
    )


def test_runtime_attestation_hash_is_recomputed_before_manifest_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _subject()
    root = tmp_path / "controlled_mvtec"
    snapshot, _, _ = _collect(module, monkeypatch, root, _complete_records(root))
    attestation = _controlled_attestation()
    object.__setattr__(attestation, "attestation_sha256", "0" * 64)
    _assert_b2_error(
        module,
        "B2_RUNTIME_ATTESTATION_INVALID",
        lambda: module.build_split_manifest(
            specification=_specification(),
            source_snapshot=snapshot,
            runtime_attestation=attestation,
            repository_identity=_repository_identity(),
            run_metadata=_run_metadata(),
        ),
    )


def test_builder_rejects_unsealed_execution_profile_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _subject()
    root = tmp_path / "controlled_mvtec"
    snapshot, _, _ = _collect(module, monkeypatch, root, _complete_records(root))
    monkeypatch.undo()

    from rad.runtime.execution_profile import ExecutionProfileAttestation

    canonical = {
        "schema_version": 1,
        "profile": {
            "profile_id": "frozen_deterministic_math",
            "path": str(PROFILE_PATH.resolve()),
            "expected_sha256": EXPECTED_PROFILE_SHA256,
            "launcher_sha256": EXPECTED_PROFILE_SHA256,
            "runtime_sha256": EXPECTED_PROFILE_SHA256,
            "hashes_match": True,
        },
        "requested_settings": {},
        "effective_settings": {},
        "environment": {},
        "canary": {
            "self_repeatability": True,
            "independent_reconstruction": True,
        },
    }
    forged = object.__new__(ExecutionProfileAttestation)
    object.__setattr__(forged, "_canonical", canonical)
    object.__setattr__(
        forged,
        "attestation_sha256",
        _canonical_sha256(canonical),
    )
    object.__setattr__(forged, "requested_settings", {})
    object.__setattr__(forged, "effective_settings", {})

    _assert_b2_error(
        module,
        "B2_RUNTIME_ATTESTATION_REQUIRED",
        lambda: module.build_split_manifest(
            specification=_specification(),
            source_snapshot=snapshot,
            runtime_attestation=forged,
            repository_identity=_repository_identity(),
            run_metadata=_run_metadata(),
        ),
    )


def test_stable_ids_and_selection_do_not_depend_on_filesystem_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _subject()
    root = tmp_path / "controlled_mvtec"
    records = list(_complete_records(root))
    reversed_result = _build(module, monkeypatch, root, list(reversed(records)))
    random.Random(93841).shuffle(records)
    shuffled_result = _build(module, monkeypatch, root, records)
    assert reversed_result.scientific_sha256 == shuffled_result.scientific_sha256
    assert module.canonical_scientific_content(_manifest(reversed_result)) == (
        module.canonical_scientific_content(_manifest(shuffled_result))
    )


def test_stable_ids_and_source_list_hash_cover_all_enumerated_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _subject()
    root = tmp_path / "controlled_mvtec"
    snapshot, _, _ = _collect(module, monkeypatch, root, _complete_records(root))
    expected_records: list[dict[str, Any]] = []
    for record in _complete_records(root):
        image_identity = record.image_path.relative_to(root).as_posix()
        anomaly_type = record.image_path.parent.name
        stable_identity = {
            "dataset": "mvtec",
            "category": record.category,
            "source_split": record.split,
            "anomaly_type": anomaly_type,
            "image_identity": image_identity,
        }
        expected_records.append(
            {
                **stable_identity,
                "image_label": record.image_label,
                "mask_identity": (
                    record.mask_path.relative_to(root).as_posix()
                    if record.mask_path is not None
                    else None
                ),
                "stable_sample_id": _canonical_sha256(stable_identity),
            }
        )
    expected_records.sort(key=lambda item: item["stable_sample_id"])
    assert list(snapshot.canonical_records) == expected_records
    assert snapshot.source_list_sha256 == _canonical_sha256(expected_records)
    assert len(snapshot.canonical_records) == 40


def test_unselected_source_record_change_changes_source_and_scientific_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _subject()
    root = tmp_path / "controlled_mvtec"
    records = list(_complete_records(root))
    first = _build(module, monkeypatch, root, records)
    changed = list(records)
    changed[-1] = _record(
        root,
        category="carpet",
        label=1,
        index=99,
    )
    second = _build(module, monkeypatch, root, changed)
    assert _manifest(first)["source"]["source_list_sha256"] != (
        _manifest(second)["source"]["source_list_sha256"]
    )
    assert first.scientific_sha256 != second.scientific_sha256


def test_same_seed_and_source_have_stable_canonical_content_across_run_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _subject()
    root = tmp_path / "controlled_mvtec"
    first = _build(module, monkeypatch, root, run_suffix="a")
    second = _build(module, monkeypatch, root, run_suffix="b")
    assert _manifest(first) != _manifest(second)
    assert first.scientific_sha256 == second.scientific_sha256
    assert module.canonical_scientific_content(_manifest(first)) == (
        module.canonical_scientific_content(_manifest(second))
    )


def test_manifest_records_source_spec_profile_base_and_worktree_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _subject()
    result = _build(module, monkeypatch, tmp_path / "controlled_mvtec")
    manifest = _manifest(result)
    assert set(manifest) == {
        "schema_version",
        "status",
        "run_id",
        "creation_timestamp",
        "output_directory",
        "git_commit",
        "branch",
        "base",
        "worktree",
        "transfer_direction",
        "source",
        "forbidden_target_dataset",
        "categories",
        "seed",
        "specification",
        "execution_profile",
        "splits",
        "count_audit",
        "overlap_audit",
        "source_only_audit",
        "fixture_path_audit",
        "mask_audit",
    }
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "passed"
    assert manifest["git_commit"] == EXPECTED_BASE_COMMIT
    assert manifest["branch"] == "phase-b2-tiny-gate-c"
    assert manifest["transfer_direction"] == "mvtec_to_visa"
    assert manifest["forbidden_target_dataset"] == "visa"
    assert manifest["categories"] == ["bottle", "carpet"]
    assert manifest["seed"] == 111
    assert manifest["source"] == {
        "dataset": "mvtec",
        "adapter_module": "rad.data.adapters.mvtec",
        "adapter_class": "MVTecAdapter",
        "record_count": 40,
        "source_list_sha256": manifest["source"]["source_list_sha256"],
        "dataset_root_identity_sha256": manifest["source"][
            "dataset_root_identity_sha256"
        ],
    }
    assert len(manifest["source"]["source_list_sha256"]) == 64
    assert len(manifest["source"]["dataset_root_identity_sha256"]) == 64
    assert manifest["specification"] == {
        "specification_id": "b2_tiny_gate_c",
        "path": "configs/phase_b/b2_tiny_gate_c.json",
        "sha256": hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(),
    }
    assert manifest["execution_profile"] == {
        "execution_profile_name": "frozen_deterministic_math",
        "execution_profile_sha256": EXPECTED_PROFILE_SHA256,
        "runtime_attestation_sha256": _controlled_attestation().attestation_sha256,
    }
    assert manifest["base"] == {
        "tag": "b1-strict-independent-v1",
        "commit": EXPECTED_BASE_COMMIT,
    }
    assert manifest["worktree"] == {
        "path": "/root/autodl-tmp/AD-phase-b2-gate-c",
        "branch": "phase-b2-tiny-gate-c",
        "git_sha": EXPECTED_BASE_COMMIT,
    }
    selected = [
        sample
        for samples in _samples_by_split(manifest).values()
        for sample in samples
    ]
    assert len(selected) == 32
    assert all(
        set(sample)
        == {
            "stable_sample_id",
            "category",
            "image_label",
            "anomaly_type",
            "image_identity",
            "mask_identity",
            "membership",
        }
        for sample in selected
    )
    assert all(not Path(sample["image_identity"]).is_absolute() for sample in selected)
    assert manifest["count_audit"] == {
        "passed": True,
        "total_selected": 32,
        "split_totals": EXPECTED_SPLIT_COUNTS,
        "by_split": {
            split: {
                category: {
                    "total": EXPECTED_PER_CATEGORY[split],
                    "normal": EXPECTED_PER_CATEGORY_LABEL[split],
                    "anomalous": EXPECTED_PER_CATEGORY_LABEL[split],
                }
                for category in ("bottle", "carpet")
            }
            for split in ("training", "calibration", "evaluation")
        },
    }
    assert manifest["overlap_audit"] == {
        "passed": True,
        "pairwise_disjoint": True,
        "unique_selected_ids": 32,
        "intersections": {
            "training_calibration": [],
            "training_evaluation": [],
            "calibration_evaluation": [],
        },
    }
    assert manifest["source_only_audit"] == {
        "passed": True,
        "source_dataset": "mvtec",
        "forbidden_target_dataset": "visa",
        "forbidden_target_access_count": 0,
    }
    assert manifest["fixture_path_audit"] == {
        "passed": True,
        "forbidden_components": ["examples", "fixtures", "synthetic", "tests"],
        "violations": [],
    }
    assert manifest["mask_audit"] == {
        "passed": True,
        "anomalous_selected": 16,
        "anomalous_with_masks": 16,
        "normal_selected": 16,
        "normal_with_masks": 0,
    }


def test_every_manifest_sample_is_bound_to_enumerated_source_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _subject()
    root = tmp_path / "controlled_mvtec"
    snapshot, _, _ = _collect(module, monkeypatch, root, _complete_records(root))
    result = module.build_split_manifest(
        specification=_specification(),
        source_snapshot=snapshot,
        runtime_attestation=_controlled_attestation(),
        repository_identity=_repository_identity(),
        run_metadata=_run_metadata(),
    )
    manifest = _manifest(result)
    enumerated = {
        record["stable_sample_id"]: record for record in snapshot.canonical_records
    }
    for split, samples in _samples_by_split(manifest).items():
        for sample in samples:
            source = enumerated[sample["stable_sample_id"]]
            assert sample["membership"] == split
            assert sample["category"] == source["category"]
            assert sample["image_label"] == source["image_label"]
            assert sample["anomaly_type"] == source["anomaly_type"]
            assert sample["image_identity"] == source["image_identity"]
            assert sample["mask_identity"] == source["mask_identity"]


def test_manifest_construction_is_pure_after_source_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _subject()
    root = tmp_path / "controlled_mvtec"
    snapshot, _, _ = _collect(module, monkeypatch, root, _complete_records(root))
    specification = _specification()

    def forbidden_io(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("pure manifest construction performed filesystem I/O")

    monkeypatch.setattr(Path, "open", forbidden_io)
    monkeypatch.setattr(Path, "read_bytes", forbidden_io)
    monkeypatch.setattr(Path, "read_text", forbidden_io)
    monkeypatch.setattr(Path, "write_bytes", forbidden_io)
    monkeypatch.setattr(Path, "write_text", forbidden_io)
    monkeypatch.setattr(Path, "iterdir", forbidden_io)
    result = module.build_split_manifest(
        specification=specification,
        source_snapshot=snapshot,
        runtime_attestation=_controlled_attestation(),
        repository_identity=_repository_identity(),
        run_metadata=_run_metadata(),
    )
    assert result.scientific_sha256


def test_canonical_scientific_hash_excludes_exactly_run_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _subject()
    manifest = dict(
        _manifest(_build(module, monkeypatch, tmp_path / "controlled_mvtec"))
    )
    canonical = module.canonical_scientific_content(manifest)
    assert {"run_id", "creation_timestamp", "output_directory"}.isdisjoint(canonical)
    expected = copy.deepcopy(manifest)
    for key in ("run_id", "creation_timestamp", "output_directory"):
        expected.pop(key)
    assert canonical == expected
    digest = hashlib.sha256(
        json.dumps(
            expected,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert module.canonical_scientific_sha256(manifest) == digest

    for key in ("run_id", "creation_timestamp", "output_directory"):
        changed = copy.deepcopy(manifest)
        changed[key] = f"changed-{key}"
        assert module.canonical_scientific_sha256(changed) == digest


@pytest.mark.parametrize(
    ("scientific_field", "mutate"),
    [
        ("sample IDs", lambda m: m["splits"]["training"][0].update(stable_sample_id="0" * 64)),
        ("memberships", lambda m: m["splits"]["training"][0].update(membership="evaluation")),
        ("categories", lambda m: m["splits"]["training"][0].update(category="changed")),
        ("labels", lambda m: m["splits"]["training"][0].update(image_label=9)),
        ("mask identities", lambda m: m["splits"]["training"][0].update(mask_identity="changed")),
        ("dataset identity", lambda m: m["source"].update(dataset="changed")),
        ("specification hash", lambda m: m["specification"].update(sha256="0" * 64)),
        ("source-list hash", lambda m: m["source"].update(source_list_sha256="0" * 64)),
        (
            "execution-profile hash",
            lambda m: m["execution_profile"].update(execution_profile_sha256="0" * 64),
        ),
        ("base identity", lambda m: m["base"].update(commit="0" * 40)),
        ("worktree identity", lambda m: m["worktree"].update(git_sha="0" * 40)),
    ],
)
def test_canonical_hash_includes_every_required_scientific_field(
    scientific_field: str,
    mutate: Callable[[dict[str, Any]], None],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _subject()
    manifest = copy.deepcopy(
        dict(_manifest(_build(module, monkeypatch, tmp_path / "controlled_mvtec")))
    )
    before = module.canonical_scientific_sha256(manifest)
    mutate(manifest)
    assert module.canonical_scientific_sha256(manifest) != before, scientific_field

"""Deterministic CPU teacher-cache fixture builder for B2-03A descriptor-artifacts tests.

Test-only. This module MUST NOT import ``rad.phase_b.b2_descriptor_artifacts`` (the
production module under test) so it keeps working before that module exists (RED)
and after (GREEN). It only reuses the already-approved, already-tested
``rad.phase_b.b2_teacher_cache`` machinery (Option A persistence, descriptor
reconstruction, cumulative-map fusion, exit-signal scoring) to build a *complete*,
scientifically self-consistent 32-sample teacher cache under ``tmp_path`` that is
explicitly marked as a non-production test fixture.

Do not import this module from production code.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

import rad.phase_b.b2_teacher_cache as cache_mod
from tests.rad.b2_hermetic import load_b2_split_fixture, write_hermetic_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
TEACHER_CACHE_CONFIG_PATH = (
    REPO_ROOT / "configs" / "phase_b" / "b2_teacher_cache_gate_c.json"
)
DESCRIPTOR_CONFIG_PATH = (
    REPO_ROOT / "configs" / "phase_b" / "b2_descriptor_artifacts_gate_c.json"
)
MAP_DIMS = ("batch", "channel", "height", "width")
FIXTURE_MAP_SHAPE = (1, 1, 4, 5)


def _deterministic_value(stable_sample_id: str, identity: cache_mod.MapIdentity) -> float:
    """Small, finite, per-(sample, map-identity) deterministic value; no RNG."""

    digest = hashlib.sha256(
        f"{stable_sample_id}:{identity.checkpoint_depth}:{identity.candidate_layer_id}".encode()
    ).digest()
    raw = int.from_bytes(digest[:4], "big")
    return 1.0 + (raw % 1000) / 100.0  # bounded to [1.0, 11.0), always finite


class FixtureTeacher:
    """Deterministic CPU-only per-sample synthetic teacher; never a production factory."""

    artifact_kind = "test_fixture"

    def __init__(
        self,
        candidate_layers: tuple[int, ...],
        prediction_depths: tuple[int, ...],
        *,
        value_fn: Any = _deterministic_value,
    ) -> None:
        self.lattice = cache_mod.expected_lattice(candidate_layers, prediction_depths)
        self._value_fn = value_fn

    def forward(self, sample: cache_mod.PlannedSample) -> cache_mod.TeacherOutput:
        maps: dict[cache_mod.MapIdentity, torch.Tensor] = {}
        for identity in self.lattice:
            value = self._value_fn(sample.stable_sample_id, identity)
            maps[identity] = torch.full(FIXTURE_MAP_SHAPE, value, dtype=torch.float32)
        mask = (
            torch.ones(FIXTURE_MAP_SHAPE, dtype=torch.float32)
            if sample.image_label == 1
            else None
        )
        return cache_mod.TeacherOutput(
            sample_id=sample.stable_sample_id,
            image_label=sample.image_label,
            anomalous_mask=mask,
            maps=maps,
            map_dimension_semantics={identity: MAP_DIMS for identity in maps},
            descriptor_source_identities=frozenset(maps),
            artifact_kind=self.artifact_kind,
        )


def build_descriptor_test_fixture(
    tmp_path: Path,
    *,
    value_fn: Any = _deterministic_value,
) -> dict[str, Any]:
    """Build a complete, deterministic 32-sample test-fixture teacher cache.

    Returns a dict with:
      - ``cache_root``: Path to the run directory containing ``samples/*.pt``.
      - ``manifest``: the final (status="passed") teacher-cache manifest, extended
        with ``artifact_kind": "test_fixture"`` and ``"eligible_for_evaluation": False``.
      - ``teacher_cache_config``: the ``TeacherCacheConfig`` used to build the cache
        (hermetic checkpoint path/hash substituted).
      - ``plan``: the ordered tuple of 32 ``PlannedSample`` (training, then
        calibration, then evaluation -- mirrors ``build_generation_plan`` order).
      - ``membership_by_id``: dict stable_sample_id -> membership.
      - ``entries``: tuple of ``PersistedSampleEntry`` (one per planned sample).
      - ``descriptor_contract``: the teacher-cache descriptor contract mapping.
      - ``teacher_cache_scientific_sha256`` / ``sample_coverage_sha256``: the exact
        hashes computed the same way as ``build_final_manifest`` for this fixture.
      - ``split_manifest``: the accepted V2 split manifest used to build the plan.
    """

    split_manifest = load_b2_split_fixture()
    base_config = cache_mod.load_teacher_cache_config(TEACHER_CACHE_CONFIG_PATH)
    checkpoint_path, checkpoint_sha256 = write_hermetic_checkpoint(
        tmp_path / "hermetic_checkpoint.pth"
    )
    config = replace(
        base_config,
        checkpoint_path=checkpoint_path,
    )
    plan = cache_mod.build_generation_plan(split_manifest, config)
    descriptor_contract = cache_mod.descriptor_contract(config, REPO_ROOT)

    teacher = FixtureTeacher(config.candidate_layers, config.prediction_depths, value_fn=value_fn)
    run_dir = tmp_path / "teacher_cache_fixture"
    (run_dir / "samples").mkdir(parents=True)

    expected_ids = frozenset(row.stable_sample_id for row in plan)
    membership_by_id: dict[str, str] = {}
    entries: list[cache_mod.PersistedSampleEntry] = []
    tensor_shapes: list[list[int]] = []
    for sample in plan:
        membership_by_id[sample.stable_sample_id] = sample.membership
        output = teacher.forward(sample)
        first_shape = tuple(int(size) for size in next(iter(output.maps.values())).shape)
        contract = cache_mod.CacheContract(
            candidate_layers=config.candidate_layers,
            prediction_depths=config.prediction_depths,
            backbone_depth=max(config.candidate_layers),
            expected_sample_ids=expected_ids,
            map_shape=first_shape,
            map_dimension_semantics=MAP_DIMS,
            production_mode=False,
        )
        validated = cache_mod.validate_teacher_output(output, contract)
        cumulative = cache_mod.build_cumulative_maps(validated)
        score = cache_mod.compute_final_image_score(cumulative[max(config.prediction_depths)])
        record = cache_mod.build_scientific_record(
            sample=sample,
            validated=validated,
            cumulative=cumulative,
            image_score=score,
            config=config,
            descriptor=descriptor_contract,
        )
        destination = run_dir / cache_mod.sample_relative_path(sample.stable_sample_id)
        entry = cache_mod.write_sample_atomic(destination, record)
        entries.append(entry)
        tensor_shapes.append(list(first_shape))

    partial = {
        "schema_version": 1,
        "status": "partial",
        "artifact_schema_version": 1,
        "artifact_kind": "test_fixture",
        "eligible_for_evaluation": False,
        "split_scientific_hash_version": config.split_scientific_hash_version,
        "split_scientific_sha256": config.split_scientific_sha256,
        "checkpoint_sha256": config.checkpoint_sha256,
        "execution_profile_name": config.execution_profile_name,
        "execution_profile_sha256": config.execution_profile_sha256,
        "runtime_attestation_sha256": "f" * 64,
        "descriptor_contract": dict(descriptor_contract),
        "planned_stable_sample_ids": [row.stable_sample_id for row in plan],
        "samples": [],
        "candidate_layers": list(config.candidate_layers),
        "prediction_depths": list(config.prediction_depths),
        "tensor_shapes": tensor_shapes,
    }
    final = cache_mod.build_final_manifest(
        partial_manifest=partial, entries=entries, plan=plan, run_dir=run_dir
    )
    manifest = dict(final)
    manifest["artifact_kind"] = "test_fixture"
    manifest["eligible_for_evaluation"] = False

    return {
        "cache_root": run_dir,
        "manifest": manifest,
        "teacher_cache_config": config,
        "plan": plan,
        "membership_by_id": membership_by_id,
        "entries": tuple(entries),
        "descriptor_contract": descriptor_contract,
        "teacher_cache_scientific_sha256": manifest["cache_scientific_sha256"],
        "sample_coverage_sha256": manifest["sample_coverage_sha256"],
        "split_manifest": split_manifest,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": config.checkpoint_sha256,
    }


def load_sample_payload(cache_root: Path, stable_sample_id: str) -> dict[str, Any]:
    """Load a persisted Option A ``.pt`` payload (scientific_record + hash)."""

    path = Path(cache_root) / cache_mod.sample_relative_path(stable_sample_id)
    return torch.load(path, map_location="cpu", weights_only=True)


def load_sample_record(cache_root: Path, stable_sample_id: str) -> dict[str, Any]:
    """Load only the persisted scientific record for one sample."""

    return dict(load_sample_payload(cache_root, stable_sample_id)["scientific_record"])


def overwrite_sample_payload(
    cache_root: Path, stable_sample_id: str, payload: dict[str, Any]
) -> None:
    """Directly overwrite a sample ``.pt`` file bypassing atomic write protections.

    Test-only: used to synthesize corrupted / digest-only / hash-mismatched
    artifacts for negative-path integrity tests.
    """

    path = Path(cache_root) / cache_mod.sample_relative_path(stable_sample_id)
    path.unlink(missing_ok=True)
    torch.save(payload, path)


def append_bytes_to_sample_file(cache_root: Path, stable_sample_id: str) -> None:
    """Append trailing bytes so the file-byte hash drifts from the manifest claim."""

    path = Path(cache_root) / cache_mod.sample_relative_path(stable_sample_id)
    with path.open("ab") as handle:
        handle.write(b"\x00corruption-marker")


def rewrite_sample_record(
    fixture: dict[str, Any],
    manifest: dict[str, Any],
    stable_sample_id: str,
    mutate_fn: Any,
) -> None:
    """Mutate one persisted sample record in-place (self-consistent) and resync
    the corresponding manifest sample entry's claimed hashes.

    ``mutate_fn`` receives a deep-copied scientific record dict (tensor values
    inline) and must return the mutated record. The record is re-hashed via the
    real ``rad.phase_b.b2_teacher_cache.record_scientific_sha256`` (so the
    persisted artifact and its own embedded hash stay internally consistent) and
    both the ``.pt`` file and the matching ``manifest["samples"]`` entry are
    updated so that ONLY the semantic content targeted by ``mutate_fn`` differs
    from the original, valid, accepted teacher cache -- isolating exactly one
    contract violation per test.
    """

    cache_root = fixture["cache_root"]
    record = load_sample_record(cache_root, stable_sample_id)
    mutated = mutate_fn(copy.deepcopy(record))
    scientific_digest = cache_mod.record_scientific_sha256(mutated)
    persisted = cache_mod._persistable_scientific_record(mutated)  # noqa: SLF001
    payload = {
        "scientific_record": persisted,
        "record_scientific_sha256": scientific_digest,
    }
    overwrite_sample_payload(cache_root, stable_sample_id, payload)
    path = Path(cache_root) / cache_mod.sample_relative_path(stable_sample_id)
    file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for entry in manifest["samples"]:
        if entry["stable_sample_id"] == stable_sample_id:
            entry["record_scientific_sha256"] = scientific_digest
            entry["record_file_sha256"] = file_digest
            break
    else:
        raise AssertionError(f"sample {stable_sample_id} not found in manifest")


def training_ids(fixture: dict[str, Any]) -> list[str]:
    return [
        row.stable_sample_id
        for row in fixture["plan"]
        if row.membership == "training"
    ]


def write_descriptor_config_json(
    tmp_path: Path,
    fixture: dict[str, Any],
    *,
    filename: str = "descriptor_config.json",
    overrides: dict[str, Any] | None = None,
) -> Path:
    """Write the tracked Gate-C descriptor-artifacts config with fixture hashes spliced in.

    Loads the real, tracked ``configs/phase_b/b2_descriptor_artifacts_gate_c.json``
    and only replaces the two fixture-dependent expected hashes
    (``expected_teacher_cache_scientific_sha256`` / ``expected_sample_coverage_sha256``)
    so tests can exercise the exact approved contract against a hermetic cache.
    """

    raw = json.loads(DESCRIPTOR_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["expected_teacher_cache_scientific_sha256"] = fixture["teacher_cache_scientific_sha256"]
    raw["expected_sample_coverage_sha256"] = fixture["sample_coverage_sha256"]
    if overrides:
        raw.update(overrides)
    path = Path(tmp_path) / filename
    path.write_text(json.dumps(raw, sort_keys=True, indent=2), encoding="utf-8")
    return path


def deep_copy_manifest(fixture: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(fixture["manifest"])


def production_like_manifest(fixture: dict[str, Any]) -> dict[str, Any]:
    """A copy of the fixture manifest with the test-only markers stripped.

    ``build_descriptor_test_fixture`` always marks its manifest with
    ``artifact_kind: "test_fixture"`` / ``eligible_for_evaluation: false`` so it can
    never be silently mistaken for a production cache. Some tests (e.g. CLI
    happy-path dry-run tests) need to exercise the *default* production
    acceptance path (``allow_test_fixture=False``), which only works once those
    two test-only marker keys are removed -- exactly what a real production
    ``build_final_manifest`` output would look like (it never sets them).
    """

    manifest = copy.deepcopy(fixture["manifest"])
    manifest.pop("artifact_kind", None)
    manifest.pop("eligible_for_evaluation", None)
    return manifest

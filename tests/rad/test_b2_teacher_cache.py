"""Task 1 contract tests for B2-02A teacher-cache planning and provenance."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import struct
import subprocess
import sys
import warnings
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
import torch

import rad.phase_b.b2_teacher_cache as subject
from rad.inference.adaptive_engine import compute_exit_signals as production_exit_signals
from rad.models.descriptors import LayerDescriptorExtractor
from rad.models.dlcm import sum_preserving_fusion as production_fusion

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs" / "phase_b" / "b2_teacher_cache_gate_c.json"
SPLIT_V2 = "91570da1fed6d7859d407196b10403581832ae0ff677a1ea7657ca76b91471f0"
SPLIT_V1 = "0b9371deb6c55f359a14959c8b46ff50205191b1189a48ee380eafaf28c5791a"
PROFILE_SHA256 = "7af8dba39633743da0380fef9710940cded655f68c9efa8f84f5a52aeddb3c8d"
CHECKPOINT_SHA256 = "97bd461163efb96e36cddb1c3adf677e4c4fc2daabb2521021689f30e799b4f4"
B2_COMMIT = "18bac047227754c975b23b46842458a5b41d5e2a"
MAP_DIMS = ("batch", "channel", "height", "width")


class FakeTeacher:
    """Deterministic CPU-only Task 2 fixture; never a production factory."""

    artifact_kind = "test_fixture"

    def __init__(
        self,
        identities: frozenset[subject.MapIdentity],
        *,
        missing: subject.MapIdentity | None = None,
        nonfinite: tuple[subject.MapIdentity, float] | None = None,
    ) -> None:
        self.identities = identities
        self.missing = missing
        self.nonfinite = nonfinite

    def forward(self, sample_id: str = "sample-1") -> subject.TeacherOutput:
        maps = {
            identity: torch.arange(20, dtype=torch.float32).reshape(1, 1, 4, 5)
            + float(identity.checkpoint_depth + identity.candidate_layer_id)
            for identity in self.identities
            if identity != self.missing
        }
        if self.nonfinite is not None:
            identity, value = self.nonfinite
            maps[identity] = maps[identity].clone()
            maps[identity][0, 0, 0, 0] = value
        return subject.TeacherOutput(
            sample_id=sample_id,
            image_label=1,
            anomalous_mask=torch.ones(1, 1, 4, 5, dtype=torch.float32),
            maps=maps,
            map_dimension_semantics={identity: MAP_DIMS for identity in maps},
            descriptor_source_identities=frozenset(maps),
            artifact_kind=self.artifact_kind,
        )


def _cache_contract(
    *,
    production_mode: bool = False,
    expected_sample_ids: frozenset[str] = frozenset({"sample-1"}),
) -> subject.CacheContract:
    return subject.CacheContract(
        candidate_layers=(6, 12, 18, 24),
        prediction_depths=(12, 18, 24),
        backbone_depth=24,
        expected_sample_ids=expected_sample_ids,
        map_shape=(1, 1, 4, 5),
        map_dimension_semantics=MAP_DIMS,
        production_mode=production_mode,
    )


def _validated_fixture() -> subject.ValidatedTeacherOutput:
    lattice = subject.expected_lattice((6, 12, 18, 24), (12, 18, 24))
    return subject.validate_teacher_output(
        FakeTeacher(lattice).forward(),
        _cache_contract(),
    )


def _config() -> subject.TeacherCacheConfig:
    return subject.load_teacher_cache_config(CONFIG)


def _rows() -> dict[str, list[dict[str, Any]]]:
    counts = {"training": 16, "calibration": 8, "evaluation": 8}
    index = 0
    result: dict[str, list[dict[str, Any]]] = {}
    for membership, count in counts.items():
        result[membership] = []
        for _ in range(count):
            anomalous = index % 2 == 1
            category = "bottle" if index % 4 < 2 else "carpet"
            anomaly_type = "crack" if anomalous else "good"
            image_identity = f"{category}/test/{anomaly_type}/{index:03d}.png"
            stable_identity = {
                "dataset": "mvtec",
                "category": category,
                "source_split": "test",
                "anomaly_type": anomaly_type,
                "image_identity": image_identity,
            }
            stable_sample_id = hashlib.sha256(
                json.dumps(
                    stable_identity,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            result[membership].append(
                {
                    "stable_sample_id": stable_sample_id,
                    "category": category,
                    "image_label": int(anomalous),
                    "anomaly_type": anomaly_type,
                    "image_identity": image_identity,
                    "mask_identity": (
                        f"{category}/ground_truth/{anomaly_type}/{index:03d}_mask.png"
                        if anomalous
                        else None
                    ),
                    "membership": membership,
                }
            )
            index += 1
    return result


def _manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "passed",
        "b1_base_tag": "b1-strict-independent-v1",
        "b1_base_commit": "3a751b2784a50eb0a08ed49e1db2df0b53608ccc",
        "transfer_direction": "mvtec_to_visa",
        "forbidden_target_dataset": "visa",
        "categories": ["bottle", "carpet"],
        "seed": 111,
        "source": {
            "dataset": "mvtec",
            "source_list_sha256": "1" * 64,
        },
        "specification": {"sha256": "2" * 64},
        "execution_profile": {"execution_profile_sha256": PROFILE_SHA256},
        "source_only_audit": {
            "passed": True,
            "forbidden_target_access_count": 0,
        },
        "scientific_hash_contract": {
            "active_version": 2,
            "legacy_canonical_hash_v1": SPLIT_V1,
            "canonical_scientific_hash_v2": SPLIT_V2,
        },
        "splits": _rows(),
    }


@pytest.fixture
def accepted_manifest() -> dict[str, Any]:
    manifest = _manifest()
    digest = subject._split_scientific_sha256(manifest)
    manifest["scientific_hash_contract"]["canonical_scientific_hash_v2"] = digest
    return manifest


@pytest.fixture
def accepted_config(accepted_manifest: dict[str, Any]) -> subject.TeacherCacheConfig:
    return replace(
        _config(),
        split_scientific_sha256=accepted_manifest["scientific_hash_contract"][
            "canonical_scientific_hash_v2"
        ],
    )


def test_tracked_configuration_pins_task_1_contract_without_sample_ids() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    config = _config()
    assert config.candidate_layers == (6, 12, 18, 24)
    assert config.prediction_depths == (12, 18, 24)
    assert config.membership_counts == MappingProxyType(
        {"training": 16, "calibration": 8, "evaluation": 8}
    )
    assert config.split_scientific_hash_version == 2
    assert config.split_scientific_sha256 == SPLIT_V2
    assert config.checkpoint_sha256 == CHECKPOINT_SHA256
    assert config.execution_profile_sha256 == PROFILE_SHA256
    assert config.b2_base_commit == B2_COMMIT
    assert raw["contracts"]["cache_tensor_contract_version"] == 1
    assert raw["contracts"]["descriptor_contract_version"] == 1
    assert raw["contracts"]["record_hash_schema_version"] == 1
    assert raw["resume_policy"]["replace_invalid_records"] is False
    assert "selected_sample_ids" not in CONFIG.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw.update(artifact_schema_version=2),
        lambda raw: raw.update(transfer_direction="visa_to_mvtec"),
        lambda raw: raw.update(candidate_layers=[6, 12, 24]),
        lambda raw: raw.update(prediction_depths=[18, 24]),
        lambda raw: raw["split"].update(specification_id="wrong"),
        lambda raw: raw["checkpoint"].update(path="/tmp/checkpoint.pth"),
        lambda raw: raw["execution_profile"].update(path="wrong.json"),
        lambda raw: raw["contracts"].update(cache_tensor_contract_version=2),
        lambda raw: raw["contracts"].update(descriptor_contract_version=2),
        lambda raw: raw["contracts"].update(
            descriptor_implementation_sha256="0" * 64
        ),
        lambda raw: raw["contracts"].update(
            descriptor_source_tensor_kind="raw_tokens"
        ),
        lambda raw: raw["contracts"].update(record_hash_schema_version=2),
        lambda raw: raw["resume_policy"].update(require_exact_plan=False),
        lambda raw: raw["resume_policy"].update(replace_invalid_records=True),
        lambda raw: raw.update(fail_closed_requirements=[]),
    ],
)
def test_tracked_configuration_rejects_all_fixed_field_drift(
    tmp_path: Path, mutate: Any
) -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    mutate(raw)
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_CONFIG_MISMATCH"):
        subject.load_teacher_cache_config(changed)


def test_builds_exact_ordered_32_sample_plan(
    accepted_manifest: dict[str, Any],
    accepted_config: subject.TeacherCacheConfig,
) -> None:
    plan = subject.build_generation_plan(accepted_manifest, accepted_config)
    assert len(plan) == 32
    assert Counter(row.membership for row in plan) == {
        "training": 16,
        "calibration": 8,
        "evaluation": 8,
    }
    expected_ids = [
        row["stable_sample_id"]
        for membership in ("training", "calibration", "evaluation")
        for row in accepted_manifest["splits"][membership]
    ]
    assert [row.stable_sample_id for row in plan] == expected_ids
    with pytest.raises((AttributeError, TypeError)):
        plan[0].membership = "evaluation"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (
            lambda manifest: manifest["scientific_hash_contract"].update(
                active_version=1,
                canonical_scientific_hash_v2=SPLIT_V1,
            ),
            "B2_CACHE_SPLIT_V2_REQUIRED",
        ),
        (
            lambda manifest: manifest["scientific_hash_contract"].update(
                canonical_scientific_hash_v2="f" * 64
            ),
            "B2_CACHE_SPLIT_HASH_MISMATCH",
        ),
        (
            lambda manifest: manifest["splits"]["training"][0].update(
                stable_sample_id="e" * 64
            ),
            "B2_CACHE_SELECTED_ID_DRIFT",
        ),
        (
            lambda manifest: manifest["splits"]["training"].append(
                copy.deepcopy(manifest["splits"]["training"][0])
            ),
            "B2_CACHE_SPLIT_COUNT_MISMATCH",
        ),
        (
            lambda manifest: manifest["splits"]["training"][0].update(
                membership="evaluation"
            ),
            "B2_CACHE_MEMBERSHIP_MISMATCH",
        ),
        (
            lambda manifest: manifest["splits"]["training"][1].update(
                mask_identity=None
            ),
            "B2_CACHE_ANOMALOUS_MASK_MISSING",
        ),
        (
            lambda manifest: manifest["source_only_audit"].update(
                forbidden_target_access_count=1
            ),
            "B2_CACHE_TARGET_ACCESS_FORBIDDEN",
        ),
    ],
)
def test_split_manifest_drift_fails_closed(
    accepted_manifest: dict[str, Any],
    accepted_config: subject.TeacherCacheConfig,
    mutation: Any,
    code: str,
) -> None:
    mutation(accepted_manifest)
    with pytest.raises(subject.TeacherCacheError, match=code):
        subject.build_generation_plan(accepted_manifest, accepted_config)


def test_duplicate_ids_fail_closed(
    accepted_manifest: dict[str, Any],
    accepted_config: subject.TeacherCacheConfig,
) -> None:
    accepted_manifest["splits"]["calibration"][0]["stable_sample_id"] = (
        accepted_manifest["splits"]["training"][0]["stable_sample_id"]
    )
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_DUPLICATE_SAMPLE"):
        subject.validate_split_manifest(accepted_manifest, accepted_config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("image_identity", ""),
        ("image_identity", "/absolute/image.png"),
        ("image_identity", "../escape.png"),
        ("image_identity", "bottle/./good/image.png"),
        ("image_identity", r"bottle\test\good\image.png"),
        ("image_identity", "tests/bottle/test/good/image.png"),
        ("image_identity", "visa/test/good/image.png"),
        ("image_identity", "target/test/good/image.png"),
        ("mask_identity", "/absolute/mask.png"),
        ("mask_identity", "../escape_mask.png"),
        ("mask_identity", r"bottle\ground_truth\crack\mask.png"),
        ("mask_identity", "fixtures/bottle/ground_truth/crack/mask.png"),
        ("mask_identity", "visa/ground_truth/crack/mask.png"),
    ],
)
def test_rejects_non_dataset_relative_posix_identities(
    accepted_manifest: dict[str, Any],
    accepted_config: subject.TeacherCacheConfig,
    field: str,
    value: str,
) -> None:
    row = accepted_manifest["splits"]["training"][1 if field == "mask_identity" else 0]
    row[field] = value
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_SAMPLE_IDENTITY_INVALID"):
        subject.validate_split_manifest(accepted_manifest, accepted_config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("category", "carpet"),
        ("anomaly_type", "crack"),
        ("image_identity", "bottle/train/good/000.png"),
        ("mask_identity", "carpet/ground_truth/crack/001_mask.png"),
        ("mask_identity", "bottle/ground_truth/hole/001_mask.png"),
    ],
)
def test_rejects_inconsistent_sample_identity_schema(
    accepted_manifest: dict[str, Any],
    accepted_config: subject.TeacherCacheConfig,
    field: str,
    value: str,
) -> None:
    row = accepted_manifest["splits"]["training"][1 if field == "mask_identity" else 0]
    row[field] = value
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_SAMPLE_IDENTITY_INVALID"):
        subject.validate_split_manifest(accepted_manifest, accepted_config)


@pytest.mark.parametrize(
    "field",
    [
        "stable_sample_id",
        "category",
        "image_label",
        "anomaly_type",
        "image_identity",
        "mask_identity",
        "membership",
    ],
)
def test_missing_sample_fields_fail_with_stable_domain_error(
    accepted_manifest: dict[str, Any],
    accepted_config: subject.TeacherCacheConfig,
    field: str,
) -> None:
    del accepted_manifest["splits"]["training"][0][field]
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_SAMPLE_SCHEMA_INVALID"):
        subject.build_generation_plan(accepted_manifest, accepted_config)


def test_unknown_sample_field_fails_with_stable_domain_error(
    accepted_manifest: dict[str, Any],
    accepted_config: subject.TeacherCacheConfig,
) -> None:
    accepted_manifest["splits"]["training"][0]["absolute_image_path"] = "/tmp/x"
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_SAMPLE_SCHEMA_INVALID"):
        subject.validate_split_manifest(accepted_manifest, accepted_config)


def test_real_scientific_hash_detects_sample_identity_and_id_drift(
    accepted_manifest: dict[str, Any],
    accepted_config: subject.TeacherCacheConfig,
) -> None:
    assert (
        subject._split_scientific_sha256(accepted_manifest)
        == accepted_config.split_scientific_sha256
    )
    for field, value in (
        ("stable_sample_id", "f" * 64),
        ("category", "carpet"),
        ("image_label", 1),
        ("mask_identity", "bottle/ground_truth/crack/000_mask.png"),
    ):
        changed = copy.deepcopy(accepted_manifest)
        changed["splits"]["training"][0][field] = value
        assert (
            subject._split_scientific_sha256(changed)
            != accepted_config.split_scientific_sha256
        )


def _outer_kwargs(
    tmp_path: Path,
    accepted_manifest: dict[str, Any],
    accepted_config: subject.TeacherCacheConfig,
    controlled_attestation: Any,
) -> dict[str, Any]:
    checkpoint = tmp_path / "epoch_2.pth"
    checkpoint.write_bytes(b"controlled checkpoint fixture")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return {
        "config": replace(
            accepted_config,
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
        ),
        "bootstrap_validated": True,
        "execution_profile_sha256": PROFILE_SHA256,
        "runtime_attestation": controlled_attestation,
        "split_manifest": accepted_manifest,
        "checkpoint_path": checkpoint,
        "b2_tag_commit": B2_COMMIT,
        "head_commit": "b" * 40,
        "head_is_descendant": True,
        "worktree_clean": True,
        "forbidden_target_access_count": 0,
    }


def test_outer_provenance_uses_legally_issued_attestation_and_real_checkpoint_hash() -> None:
    child = f"""
import copy
import hashlib
import json
import runpy
import tempfile
from dataclasses import replace
from pathlib import Path

from rad.runtime.execution_profile import apply_execution_profile

attestation = apply_execution_profile()
namespace = runpy.run_path({str(Path(__file__).resolve())!r})
subject = namespace["subject"]
manifest = namespace["_manifest"]()
digest = subject._split_scientific_sha256(manifest)
manifest["scientific_hash_contract"]["canonical_scientific_hash_v2"] = digest
config = replace(namespace["_config"](), split_scientific_sha256=digest)

def require_code(kwargs, code):
    try:
        subject.validate_outer_provenance(**kwargs)
    except subject.TeacherCacheError as exc:
        assert exc.code == code, (exc.code, code)
    else:
        raise AssertionError(f"expected {{code}}")

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    kwargs = namespace["_outer_kwargs"](root, manifest, config, attestation)
    provenance = subject.validate_outer_provenance(**kwargs)
    assert provenance.runtime_attestation is attestation.canonical_attestation()
    assert provenance.runtime_attestation_sha256 == attestation.attestation_sha256
    assert provenance.checkpoint_sha256 == hashlib.sha256(
        provenance.checkpoint_path.read_bytes()
    ).hexdigest()
    try:
        provenance.runtime_attestation["profile"] = {{}}
    except TypeError:
        pass
    else:
        raise AssertionError("canonical attestation must remain immutable")
    assert not hasattr(
        subject.build_generation_plan(manifest, config)[0],
        "runtime_attestation",
    )

    forged = dict(kwargs)
    forged["runtime_attestation"] = {{
        "attestation_sha256": attestation.attestation_sha256,
        "execution_profile_sha256": namespace["PROFILE_SHA256"],
        "immutable": True,
    }}
    require_code(forged, "B2_CACHE_RUNTIME_ATTESTATION_REQUIRED")

    matrix = [
        ({{"bootstrap_validated": False}}, "B2_CACHE_BOOTSTRAP_REQUIRED"),
        ({{"execution_profile_sha256": "0" * 64}}, "B2_CACHE_PROFILE_HASH_MISMATCH"),
        ({{"runtime_attestation": None}}, "B2_CACHE_RUNTIME_ATTESTATION_REQUIRED"),
        ({{"split_manifest": None}}, "B2_CACHE_SPLIT_REQUIRED"),
        ({{"checkpoint_path": None}}, "B2_CACHE_CHECKPOINT_MISSING"),
        ({{"b2_tag_commit": None}}, "B2_CACHE_B2_TAG_UNRESOLVED"),
        ({{"b2_tag_commit": "0" * 40}}, "B2_CACHE_B2_TAG_MOVED"),
        ({{"head_is_descendant": False}}, "B2_CACHE_HEAD_NOT_DESCENDANT"),
        ({{"worktree_clean": False}}, "B2_CACHE_WORKTREE_DIRTY"),
        ({{"forbidden_target_access_count": 1}}, "B2_CACHE_TARGET_ACCESS_FORBIDDEN"),
    ]
    for change, code in matrix:
        changed = dict(kwargs)
        changed.update(change)
        require_code(changed, code)

    missing = dict(kwargs)
    missing["checkpoint_path"] = root / "missing.pth"
    require_code(missing, "B2_CACHE_CHECKPOINT_MISSING")

    wrong_hash = dict(kwargs)
    wrong_hash["config"] = replace(kwargs["config"], checkpoint_sha256="0" * 64)
    require_code(wrong_hash, "B2_CACHE_CHECKPOINT_HASH_MISMATCH")

print(json.dumps({{"status": "passed", "cases": 14}}, sort_keys=True))
"""
    environment = dict(os.environ)
    for key in (
        "RAD_EXECUTION_PROFILE_BOOTSTRAPPED",
        "RAD_EXECUTION_PROFILE_PATH",
        "RAD_EXECUTION_PROFILE_SHA256",
        "CUBLAS_WORKSPACE_CONFIG",
    ):
        environment.pop(key, None)
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/run_with_execution_profile.py"),
            "--profile",
            str(REPO_ROOT / "configs/execution/frozen_deterministic_math.json"),
            "--expected-sha256",
            PROFILE_SHA256,
            "--",
            sys.executable,
            "-c",
            child,
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.splitlines()[-1]) == {
        "cases": 14,
        "status": "passed",
    }


def test_expected_lattice_is_exact_and_configuration_driven() -> None:
    assert subject.expected_lattice((6, 12, 18, 24), (12, 18, 24)) == {
        subject.MapIdentity(12, 6),
        subject.MapIdentity(12, 12),
        subject.MapIdentity(18, 6),
        subject.MapIdentity(18, 12),
        subject.MapIdentity(18, 18),
        subject.MapIdentity(24, 6),
        subject.MapIdentity(24, 12),
        subject.MapIdentity(24, 18),
        subject.MapIdentity(24, 24),
    }
    assert subject.expected_lattice((2, 5, 9), (5, 9)) == {
        subject.MapIdentity(5, 2),
        subject.MapIdentity(5, 5),
        subject.MapIdentity(9, 2),
        subject.MapIdentity(9, 5),
        subject.MapIdentity(9, 9),
    }


@pytest.mark.parametrize(
    ("candidate_layers", "prediction_depths", "backbone_depth"),
    [
        ((12, 6), (12,), 12),
        ((6, 6), (12,), 12),
        ((6, 12), (12, 12), 12),
        ((6, 13), (12,), 12),
        ((6, 12), (13,), 12),
    ],
)
def test_lattice_rejects_unsorted_duplicate_or_out_of_backbone_configuration(
    candidate_layers: tuple[int, ...],
    prediction_depths: tuple[int, ...],
    backbone_depth: int,
) -> None:
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_TENSOR_CONTRACT_INVALID"):
        subject.CacheContract(
            candidate_layers=candidate_layers,
            prediction_depths=prediction_depths,
            backbone_depth=backbone_depth,
            expected_sample_ids=frozenset({"sample-1"}),
            map_shape=(1, 1, 4, 5),
            map_dimension_semantics=MAP_DIMS,
        )


@pytest.mark.parametrize("variant", ["missing", "extra", "missing_descriptor"])
def test_teacher_output_requires_exact_map_and_descriptor_identity_sets(
    variant: str,
) -> None:
    lattice = subject.expected_lattice((6, 12, 18, 24), (12, 18, 24))
    output = FakeTeacher(lattice).forward()
    if variant == "missing":
        identity = min(lattice)
        output.maps.pop(identity)
        output.map_dimension_semantics.pop(identity)
        output = replace(output, descriptor_source_identities=frozenset(output.maps))
    elif variant == "extra":
        identity = subject.MapIdentity(24, 5)
        output.maps[identity] = torch.zeros(1, 1, 4, 5)
        output.map_dimension_semantics[identity] = MAP_DIMS
        output = replace(
            output,
            descriptor_source_identities=frozenset(output.maps),
        )
    else:
        output = replace(
            output,
            descriptor_source_identities=frozenset(lattice - {min(lattice)}),
        )
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_MAP_LATTICE_MISMATCH"):
        subject.validate_teacher_output(output, _cache_contract())


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (
            lambda output: output.maps.__setitem__(
                min(output.maps), torch.zeros(1, 4, 5)
            ),
            "B2_CACHE_MAP_SHAPE_INVALID",
        ),
        (
            lambda output: output.maps.__setitem__(
                min(output.maps), torch.zeros(1, 1, 4, 5, dtype=torch.float64)
            ),
            "B2_CACHE_TENSOR_DTYPE_INVALID",
        ),
        (
            lambda output: output.map_dimension_semantics.__setitem__(
                min(output.maps), ("batch", "height", "width", "channel")
            ),
            "B2_CACHE_DIMENSION_SEMANTICS_INVALID",
        ),
    ],
)
def test_teacher_output_enforces_shape_float32_and_explicit_dimensions(
    mutation: Any,
    code: str,
) -> None:
    lattice = subject.expected_lattice((6, 12, 18, 24), (12, 18, 24))
    output = FakeTeacher(lattice).forward()
    mutation(output)
    with pytest.raises(subject.TeacherCacheError, match=code):
        subject.validate_teacher_output(output, _cache_contract())


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_teacher_output_rejects_nonfinite_maps(value: float) -> None:
    lattice = subject.expected_lattice((6, 12, 18, 24), (12, 18, 24))
    identity = min(lattice)
    output = FakeTeacher(lattice, nonfinite=(identity, value)).forward()
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_TENSOR_NONFINITE"):
        subject.validate_teacher_output(output, _cache_contract())


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (
            lambda output: replace(output, anomalous_mask=None),
            "B2_CACHE_ANOMALOUS_MASK_MISSING",
        ),
        (
            lambda output: replace(output, sample_id="unexpected"),
            "B2_CACHE_UNEXPECTED_SAMPLE",
        ),
    ],
)
def test_teacher_output_rejects_missing_mask_and_unexpected_sample(
    mutation: Any,
    code: str,
) -> None:
    lattice = subject.expected_lattice((6, 12, 18, 24), (12, 18, 24))
    with pytest.raises(subject.TeacherCacheError, match=code):
        subject.validate_teacher_output(
            mutation(FakeTeacher(lattice).forward()),
            _cache_contract(),
        )


def test_contract_rejects_missing_expected_sample_set() -> None:
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_EXPECTED_SAMPLE_MISSING"):
        _cache_contract(expected_sample_ids=frozenset())


def test_cumulative_maps_directly_reuse_sum_preserving_fusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _validated_fixture()
    calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def fake_fusion(
        maps: torch.Tensor,
        weights: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        calls.append((maps, weights, valid_mask))
        return maps.sum(dim=1)

    monkeypatch.setattr(subject, "sum_preserving_fusion", fake_fusion)
    cumulative = subject.build_cumulative_maps(validated)
    assert list(cumulative) == [12, 18, 24]
    assert [call[0].shape[1] for call in calls] == [2, 3, 4]
    assert all(torch.equal(weights, mask.float() / mask.sum(1, keepdim=True)) for _, weights, mask in calls)
    assert all(mask.dtype is torch.bool and bool(mask.all()) for _, _, mask in calls)


def test_descriptor_reconstruction_directly_reuses_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _validated_fixture()
    instances: list[Any] = []

    class SpyExtractor:
        def __init__(self) -> None:
            instances.append(self)

        def __call__(
            self,
            maps: torch.Tensor,
            valid_mask: torch.Tensor,
        ) -> torch.Tensor:
            assert maps.ndim == 4
            assert valid_mask.dtype is torch.bool
            return torch.full((*maps.shape[:2], 18), 7.0)

    monkeypatch.setattr(subject, "LayerDescriptorExtractor", SpyExtractor)
    descriptors = subject.reconstruct_descriptors(validated)
    assert len(instances) == 1
    assert {depth: value.shape for depth, value in descriptors.items()} == {
        12: (1, 2, 18),
        18: (1, 3, 18),
        24: (1, 4, 18),
    }


def _extractor_equal_average_reference(maps_bl1hw: torch.Tensor) -> torch.Tensor:
    """Mirror LayerDescriptorExtractor default fused semantics (equal average)."""

    maps = maps_bl1hw.squeeze(2)
    valid_mask = torch.ones(maps.shape[:2], dtype=torch.bool, device=maps.device)
    weights = valid_mask.to(maps.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (maps * weights[:, :, None, None]).sum(dim=1, keepdim=True)


def test_image_score_reuses_exit_signals_with_sum_map_not_equal_average(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _validated_fixture()
    cumulative = subject.build_cumulative_maps(validated)
    full_depth = max(validated.prediction_depths)
    sum_preserving_map = cumulative[full_depth]
    identities = tuple(
        subject.MapIdentity(full_depth, layer)
        for layer in validated.candidate_layers
        if layer <= full_depth
    )
    stacked = torch.stack([validated.maps[identity] for identity in identities], dim=1)
    equal_average_reference = _extractor_equal_average_reference(stacked)
    assert not torch.equal(sum_preserving_map, equal_average_reference)

    received: list[torch.Tensor] = []

    class Signals:
        image_score = 0.75

    def fake_signals(
        fused: torch.Tensor,
        prev_fused: torch.Tensor | None,
    ) -> Signals:
        assert prev_fused is None
        received.append(fused)
        return Signals()

    monkeypatch.setattr(subject, "compute_exit_signals", fake_signals)
    score = subject.compute_final_image_score(sum_preserving_map)
    assert len(received) == 1
    assert received[0] is sum_preserving_map
    assert torch.equal(received[0], sum_preserving_map)
    assert not torch.equal(received[0], equal_average_reference)
    assert torch.equal(score, torch.tensor([0.75], dtype=torch.float32))


def test_production_call_boundaries_bind_to_authoritative_objects() -> None:
    subject._bind_production_tensor_apis()
    assert subject.sum_preserving_fusion is production_fusion
    assert subject.LayerDescriptorExtractor is LayerDescriptorExtractor
    assert subject.compute_exit_signals is production_exit_signals


def test_fake_teacher_is_absent_from_production_module() -> None:
    assert not hasattr(subject, "FakeTeacher")
    assert not hasattr(subject, "create_test_fixture_teacher")
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "class FakeTeacher" not in source
    assert 'artifact_kind="test_fixture"' not in source
    assert "artifact_kind='test_fixture'" not in source


def test_teacher_output_rejects_non_float32_anomalous_mask() -> None:
    lattice = subject.expected_lattice((6, 12, 18, 24), (12, 18, 24))
    output = replace(
        FakeTeacher(lattice).forward(),
        anomalous_mask=torch.ones(1, 1, 4, 5, dtype=torch.float64),
    )
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_TENSOR_DTYPE_INVALID"):
        subject.validate_teacher_output(output, _cache_contract())


def test_nonfinite_cumulative_map_and_image_score_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _validated_fixture()
    monkeypatch.setattr(
        subject,
        "sum_preserving_fusion",
        lambda maps, weights, valid_mask: torch.full(
            (1, 1, 4, 5), float("inf"), dtype=torch.float32
        ),
    )
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_TENSOR_NONFINITE"):
        subject.build_cumulative_maps(validated)

    class Signals:
        image_score = float("nan")

    monkeypatch.setattr(
        subject,
        "compute_exit_signals",
        lambda fused, prev_fused: Signals(),
    )
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_TENSOR_NONFINITE"):
        subject.compute_final_image_score(torch.zeros(1, 1, 4, 5))


@pytest.mark.parametrize(
    ("full_depth_map", "code"),
    [
        (torch.zeros(1, 4, 5), "B2_CACHE_MAP_SHAPE_INVALID"),
        (
            torch.zeros(1, 1, 4, 5, dtype=torch.float64),
            "B2_CACHE_TENSOR_DTYPE_INVALID",
        ),
    ],
)
def test_image_score_enforces_primary_tensor_contract(
    full_depth_map: torch.Tensor,
    code: str,
) -> None:
    with pytest.raises(subject.TeacherCacheError, match=code):
        subject.compute_final_image_score(full_depth_map)


def test_fake_live_forward_equals_roundtrip_cache_descriptors(tmp_path: Path) -> None:
    lattice = subject.expected_lattice((6, 12, 18, 24), (12, 18, 24))
    output = FakeTeacher(lattice).forward()
    validated_live = subject.validate_teacher_output(output, _cache_contract())
    live = {}
    extractor = LayerDescriptorExtractor()
    for depth in (12, 18, 24):
        identities = sorted(
            identity for identity in lattice if identity.checkpoint_depth == depth
        )
        maps = torch.cat([output.maps[identity] for identity in identities], dim=1)
        live[depth] = extractor(
            maps.squeeze(2),
            torch.ones(1, len(identities), dtype=torch.bool),
        )

    payload_path = tmp_path / "future-payload.pt"
    torch.save(
        {
            "sample_id": validated_live.sample_id,
            "maps": {
                (identity.checkpoint_depth, identity.candidate_layer_id): tensor
                for identity, tensor in validated_live.maps.items()
            },
            "map_dimension_semantics": {
                (identity.checkpoint_depth, identity.candidate_layer_id): semantics
                for identity, semantics in validated_live.map_dimension_semantics.items()
            },
        },
        payload_path,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="TypedStorage is deprecated",
            category=UserWarning,
        )
        loaded = torch.load(payload_path, map_location="cpu", weights_only=True)
    roundtripped = replace(
        output,
        maps={
            subject.MapIdentity(*identity): tensor
            for identity, tensor in loaded["maps"].items()
        },
        map_dimension_semantics={
            subject.MapIdentity(*identity): tuple(semantics)
            for identity, semantics in loaded["map_dimension_semantics"].items()
        },
    )
    cached = subject.reconstruct_descriptors(
        subject.validate_teacher_output(roundtripped, _cache_contract())
    )
    assert all(torch.equal(live[depth], cached[depth]) for depth in (12, 18, 24))


def test_production_mode_hard_rejects_test_fixture_teacher() -> None:
    lattice = subject.expected_lattice((6, 12, 18, 24), (12, 18, 24))
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_TEST_FIXTURE_FORBIDDEN"):
        subject.validate_teacher_output(
            FakeTeacher(lattice).forward(),
            _cache_contract(production_mode=True),
        )


# --- Task 3: versioned descriptor contract and canonical scientific hashing ---

DESCRIPTOR_IMPL_SHA256 = (
    "6846ad263d342649a0383c4f762f7820053428bf74a05ece8c02e1dcc641b615"
)
FEATURE_NAMES = (
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


def _tensor_meta(
    name: str,
    tensor: torch.Tensor,
    semantics: tuple[str, ...] = MAP_DIMS,
) -> dict[str, Any]:
    return {
        "logical_name": name,
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "shape": list(tensor.shape),
        "dimension_semantics": list(semantics),
        "digest": subject.canonical_tensor_digest(name, tensor, semantics),
    }


def _scientific_record_fixture() -> dict[str, Any]:
    lattice = subject.expected_lattice((6, 12, 18, 24), (12, 18, 24))
    tensors: dict[str, Any] = {}
    for identity in sorted(lattice):
        name = (
            f"causal_map:{identity.checkpoint_depth}:{identity.candidate_layer_id}"
        )
        tensor = torch.full(
            (1, 1, 4, 5),
            float(identity.checkpoint_depth + identity.candidate_layer_id),
            dtype=torch.float32,
        )
        tensors[name] = _tensor_meta(name, tensor)
    for depth in (12, 18, 24):
        name = f"cumulative_map:{depth}"
        tensors[name] = _tensor_meta(
            name, torch.ones(1, 1, 4, 5, dtype=torch.float32) * float(depth)
        )
    tensors["full_depth_map"] = _tensor_meta(
        "full_depth_map", torch.ones(1, 1, 4, 5, dtype=torch.float32) * 24.0
    )
    tensors["anomalous_mask"] = _tensor_meta(
        "anomalous_mask", torch.ones(1, 1, 4, 5, dtype=torch.float32)
    )
    tensors["image_score"] = _tensor_meta(
        "image_score",
        torch.tensor([0.5], dtype=torch.float32),
        ("scalar",),
    )
    contract = subject.descriptor_contract(_config(), REPO_ROOT)
    return {
        "record_schema_version": 1,
        "record_hash_schema_version": 1,
        "stable_sample_id": "a" * 64,
        "membership": "training",
        "category": "bottle",
        "image_label": 1,
        "anomaly_type": "crack",
        "image_identity": "bottle/test/crack/001.png",
        "mask_identity": "bottle/ground_truth/crack/001_mask.png",
        "candidate_layers": [6, 12, 18, 24],
        "prediction_depths": [12, 18, 24],
        "causal_map_lattice": [
            {
                "checkpoint_depth": identity.checkpoint_depth,
                "candidate_layer_id": identity.candidate_layer_id,
            }
            for identity in sorted(lattice)
        ],
        "cache_tensor_contract_version": 1,
        "tensors": tensors,
        "descriptor_contract_version": contract["descriptor_contract_version"],
        "descriptor_feature_names": list(contract["feature_names"]),
        "descriptor_source_tensor_kind": contract["descriptor_source_tensor_kind"],
        "descriptor_implementation_sha256": contract[
            "descriptor_implementation_sha256"
        ],
        "extractor_configuration_sha256": contract[
            "extractor_configuration_sha256"
        ],
        "split_scientific_hash_version": 2,
        "split_scientific_sha256": SPLIT_V2,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "execution_profile_name": "frozen_deterministic_math",
        "execution_profile_sha256": PROFILE_SHA256,
        # Excluded outer/runtime provenance (must remain present but ignored).
        "runtime_attestation_sha256": "r" * 64,
        "generation_commit": "c" * 40,
        "generation_branch": "phase-b2-teacher-cache",
        "worktree_clean": True,
        "worktree_status": "clean",
        "machine_hostname": "gpu-host",
        "environment": {"CUDA_VISIBLE_DEVICES": "0", "python": "3.10.20"},
        "run_id": "run-should-not-hash",
        "output_path": "/tmp/teacher-cache-run",
        "absolute_image_path": "/data/mvtec/bottle/test/crack/001.png",
        "absolute_mask_path": "/data/mvtec/bottle/ground_truth/crack/001_mask.png",
        "checkpoint_path": "/root/autodl-tmp/AD/runs/baseline/epoch_2.pth",
        "timestamp": "2026-07-21T00:00:00Z",
        "record_file_sha256": "f" * 64,
    }


def test_descriptor_implementation_sha256_matches_tracked_bytes() -> None:
    digest = subject.descriptor_implementation_sha256(REPO_ROOT)
    assert digest == DESCRIPTOR_IMPL_SHA256
    assert digest == hashlib.sha256(
        (REPO_ROOT / "rad" / "models" / "descriptors.py").read_bytes()
    ).hexdigest()


def test_descriptor_contract_pins_version_order_source_and_digests() -> None:
    contract = subject.descriptor_contract(_config(), REPO_ROOT)
    assert contract["descriptor_contract_version"] == 1
    assert tuple(contract["feature_names"]) == FEATURE_NAMES
    assert contract["descriptor_source_tensor_kind"] == "causal_anomaly_maps"
    assert contract["descriptor_implementation_sha256"] == DESCRIPTOR_IMPL_SHA256
    assert isinstance(contract["extractor_configuration_sha256"], str)
    assert len(contract["extractor_configuration_sha256"]) == 64
    assert all(
        character in "0123456789abcdef"
        for character in contract["extractor_configuration_sha256"]
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda contract: contract.update(descriptor_contract_version=2),
        lambda contract: contract.update(
            feature_names=list(FEATURE_NAMES[::-1])
        ),
        lambda contract: contract.update(
            descriptor_source_tensor_kind="raw_tokens"
        ),
        lambda contract: contract.update(
            descriptor_implementation_sha256="0" * 64
        ),
        lambda contract: contract.update(
            extractor_configuration_sha256="1" * 64
        ),
    ],
)
def test_descriptor_contract_drift_fails_closed(mutate: Any) -> None:
    expected = subject.descriptor_contract(_config(), REPO_ROOT)
    loaded = dict(expected)
    mutate(loaded)
    with pytest.raises(
        subject.TeacherCacheError, match="B2_CACHE_DESCRIPTOR_CONTRACT_DRIFT"
    ):
        subject.validate_descriptor_contract(loaded, expected)


def test_descriptor_contract_rejects_implementation_byte_drift(
    tmp_path: Path,
) -> None:
    config = _config()
    drifted = replace(config, descriptor_implementation_sha256="0" * 64)
    with pytest.raises(
        subject.TeacherCacheError, match="B2_CACHE_DESCRIPTOR_CONTRACT_DRIFT"
    ):
        subject.descriptor_contract(drifted, REPO_ROOT)


def test_canonical_tensor_digest_is_deterministic_and_path_time_independent() -> None:
    tensor = torch.arange(20, dtype=torch.float32).reshape(1, 1, 4, 5)
    first = subject.canonical_tensor_digest("map", tensor, MAP_DIMS)
    second = subject.canonical_tensor_digest(
        "map", tensor.clone(), MAP_DIMS
    )
    assert first == second
    assert first == subject.canonical_tensor_digest(
        "map",
        tensor,
        MAP_DIMS,
    )


def test_canonical_tensor_digests_combine_in_sorted_logical_name_order() -> None:
    record = _scientific_record_fixture()
    baseline = subject.record_scientific_sha256(record)
    # Insertion order differs, but scientific hashing sorts logical names.
    reordered = copy.deepcopy(record)
    items = list(reordered["tensors"].items())
    reordered["tensors"] = dict(reversed(items))
    assert list(reordered["tensors"]) != list(record["tensors"])
    assert subject.record_scientific_sha256(reordered) == baseline
    renamed = copy.deepcopy(record)
    first_name = next(iter(renamed["tensors"]))
    meta = renamed["tensors"].pop(first_name)
    meta = dict(meta)
    meta["logical_name"] = "zzz_" + first_name
    meta["digest"] = subject.canonical_tensor_digest(
        meta["logical_name"],
        torch.ones(1, 1, 4, 5, dtype=torch.float32),
        MAP_DIMS,
    )
    renamed["tensors"][meta["logical_name"]] = meta
    assert subject.record_scientific_sha256(renamed) != baseline


def test_canonical_tensor_digest_is_value_shape_and_dtype_sensitive() -> None:
    base = torch.arange(20, dtype=torch.float32).reshape(1, 1, 4, 5)
    digest = subject.canonical_tensor_digest("map", base, MAP_DIMS)
    value_changed = base.clone()
    value_changed[0, 0, 0, 0] = 99.0
    assert (
        subject.canonical_tensor_digest("map", value_changed, MAP_DIMS) != digest
    )
    assert (
        subject.canonical_tensor_digest(
            "map", base.reshape(1, 1, 5, 4), ("batch", "channel", "height", "width")
        )
        != digest
    )
    assert (
        subject.canonical_tensor_digest(
            "map", base.to(torch.float64), MAP_DIMS
        )
        != digest
    )
    assert (
        subject.canonical_tensor_digest("other", base, MAP_DIMS) != digest
    )
    assert (
        subject.canonical_tensor_digest(
            "map", base, ("batch", "height", "width", "channel")
        )
        != digest
    )


def _length_delimited_canonical_tensor_payload_for_test(
    name: str,
    dtype_name: str,
    shape: tuple[int, ...],
    dimension_semantics: tuple[str, ...],
    raw_bytes: bytes,
) -> bytes:
    """Mirror production length-delimited encoding without calling production hash helpers."""

    def u64(value: int) -> bytes:
        return int(value).to_bytes(8, "little", signed=False)

    def ld_bytes(payload: bytes) -> bytes:
        return u64(len(payload)) + payload

    def ld_text(value: str) -> bytes:
        return ld_bytes(value.encode("utf-8"))

    parts = bytearray()
    parts.extend(ld_text(name))
    parts.extend(ld_text(dtype_name))
    parts.extend(u64(len(shape)))
    for size in shape:
        parts.extend(u64(int(size)))
    parts.extend(u64(len(dimension_semantics)))
    for item in dimension_semantics:
        parts.extend(ld_text(item))
    parts.extend(ld_bytes(raw_bytes))
    return bytes(parts)


def test_canonical_tensor_digest_accepts_noncontiguous_and_is_little_endian() -> None:
    contiguous = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    noncontiguous = contiguous.t()
    assert not noncontiguous.is_contiguous()
    digest = subject.canonical_tensor_digest(
        "plane", noncontiguous, ("row", "col")
    )
    assert digest == subject.canonical_tensor_digest(
        "plane", contiguous.t().contiguous(), ("row", "col")
    )

    # Independent LE proof: known float32 values → explicit little-endian bytes,
    # then the same length-delimited metadata prefix as the production encoder.
    probe = torch.tensor([1.0, -2.5, 256.0], dtype=torch.float32)
    name = "probe"
    semantics = ("feature",)
    expected_le = struct.pack("<fff", 1.0, -2.5, 256.0)
    expected_be = struct.pack(">fff", 1.0, -2.5, 256.0)
    assert expected_le != expected_be
    assert expected_le == bytes.fromhex("0000803f000020c000008043")
    expected_digest = hashlib.sha256(
        _length_delimited_canonical_tensor_payload_for_test(
            name,
            "float32",
            tuple(probe.shape),
            semantics,
            expected_le,
        )
    ).hexdigest()
    be_digest = hashlib.sha256(
        _length_delimited_canonical_tensor_payload_for_test(
            name,
            "float32",
            tuple(probe.shape),
            semantics,
            expected_be,
        )
    ).hexdigest()
    actual = subject.canonical_tensor_digest(name, probe, semantics)
    assert actual == expected_digest
    assert actual != be_digest


@pytest.mark.parametrize(
    "factory",
    [
        lambda: torch.sparse_coo_tensor(
            indices=torch.tensor([[0], [0]]),
            values=torch.tensor([1.0]),
            size=(2, 2),
        ),
        lambda: torch.quantize_per_tensor(
            torch.ones(2, 2), scale=0.1, zero_point=0, dtype=torch.qint8
        ),
        lambda: torch.tensor([float("nan")], dtype=torch.float32),
        lambda: torch.tensor([float("inf")], dtype=torch.float32),
    ],
)
def test_canonical_tensor_digest_rejects_sparse_quantized_nonfinite(
    factory: Any,
) -> None:
    tensor = factory()
    with pytest.raises(
        subject.TeacherCacheError,
        match=(
            "B2_CACHE_TENSOR_UNSUPPORTED|B2_CACHE_TENSOR_NONFINITE|"
            "B2_CACHE_TENSOR_DTYPE_INVALID"
        ),
    ):
        subject.canonical_tensor_digest("bad", tensor, ("feature",) * tensor.ndim)


def test_record_scientific_hash_uses_explicit_whitelist_not_exclude_all() -> None:
    record = _scientific_record_fixture()
    content = subject.scientific_record_content(record)
    assert "record_hash_schema_version" in content
    assert content["record_hash_schema_version"] == 1
    for excluded in (
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
    ):
        assert excluded not in content
    for required in (
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
    ):
        assert required in content
    # Explicit whitelist construction: unknown scientific keys are rejected.
    poisoned = dict(record)
    poisoned["extra_scientific_field"] = "should-fail"
    with pytest.raises(
        subject.TeacherCacheError, match="B2_CACHE_RECORD_HASH_SCHEMA_INVALID"
    ):
        subject.scientific_record_content(poisoned)


@pytest.mark.parametrize(
    "field",
    [
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
        "record_schema_version",
    ],
)
def test_record_scientific_hash_is_sensitive_to_whitelist_fields(field: str) -> None:
    record = _scientific_record_fixture()
    baseline = subject.record_scientific_sha256(record)
    changed = copy.deepcopy(record)
    value = changed[field]
    if field == "causal_map_lattice":
        changed[field] = copy.deepcopy(value)
        changed[field][0]["candidate_layer_id"] = 99
    elif isinstance(value, str):
        changed[field] = ("b" if value[:1] != "b" else "c") + value[1:]
        if field.endswith("sha256") or field == "stable_sample_id":
            changed[field] = ("0" if value[:1] != "0" else "1") * 64
    elif isinstance(value, int):
        changed[field] = value + 1
    elif isinstance(value, list):
        changed[field] = list(reversed(value)) if value else [0]
    else:
        changed[field] = value
    assert subject.record_scientific_sha256(changed) != baseline, field


def test_unsupported_record_hash_schema_version_fails_closed() -> None:
    record = _scientific_record_fixture()
    record["record_hash_schema_version"] = 2
    with pytest.raises(
        subject.TeacherCacheError, match="B2_CACHE_RECORD_HASH_SCHEMA_INVALID"
    ):
        subject.record_scientific_sha256(record)


def test_record_scientific_hash_is_sensitive_to_tensor_digest() -> None:
    record = _scientific_record_fixture()
    baseline = subject.record_scientific_sha256(record)
    changed = copy.deepcopy(record)
    name = next(iter(changed["tensors"]))
    meta = changed["tensors"][name]
    meta["digest"] = "0" * 64
    assert subject.record_scientific_sha256(changed) != baseline


@pytest.mark.parametrize(
    "field",
    [
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
    ],
)
def test_record_scientific_hash_ignores_excluded_provenance_fields(
    field: str,
) -> None:
    record = _scientific_record_fixture()
    baseline = subject.record_scientific_sha256(record)
    changed = copy.deepcopy(record)
    if isinstance(changed[field], bool):
        changed[field] = not changed[field]
    elif isinstance(changed[field], dict):
        changed[field] = {"mutated": True}
    else:
        changed[field] = f"mutated-{changed[field]}"
    assert subject.record_scientific_sha256(changed) == baseline
    # Excluded operational fields may remain on the outer record envelope.
    subject.scientific_record_content(changed)


def test_record_scientific_hash_ignores_operational_paths_timestamps_and_run_id() -> None:
    left = _scientific_record_fixture()
    right = copy.deepcopy(left)
    right["output_path"] = "/other/run"
    right["timestamp"] = "2099-01-01T00:00:00Z"
    right["absolute_image_path"] = "/elsewhere/image.png"
    right["run_id"] = "different-run"
    assert subject.record_scientific_sha256(left) == subject.record_scientific_sha256(
        right
    )
    assert left["image_identity"] == right["image_identity"]
    assert "absolute_image_path" not in subject.scientific_record_content(left)
    assert "image_identity" in subject.scientific_record_content(left)


def test_primary_record_rejects_non_float32_scientific_tensors() -> None:
    record = _scientific_record_fixture()
    name = "causal_map:12:6"
    bad = torch.ones(1, 1, 4, 5, dtype=torch.float64)
    record["tensors"][name] = {
        "logical_name": name,
        "dtype": "float64",
        "shape": list(bad.shape),
        "dimension_semantics": list(MAP_DIMS),
        "digest": subject.canonical_tensor_digest(name, bad, MAP_DIMS),
    }
    with pytest.raises(
        subject.TeacherCacheError, match="B2_CACHE_TENSOR_DTYPE_INVALID"
    ):
        subject.scientific_record_content(record)


def test_record_scientific_hash_is_deterministic() -> None:
    record = _scientific_record_fixture()
    assert subject.record_scientific_sha256(record) == subject.record_scientific_sha256(
        copy.deepcopy(record)
    )


# --- Task 4: Option A atomic persistence, resume, and coverage ---


def _scientific_only(record: Mapping[str, Any]) -> dict[str, Any]:
    return dict(subject.scientific_record_content(record))


def _sample_pt_path(run_dir: Path, stable_sample_id: str) -> Path:
    return run_dir / "samples" / f"{stable_sample_id}.pt"


def _partial_manifest_skeleton(
    *,
    plan: tuple[subject.PlannedSample, ...],
    entries: list[dict[str, Any]] | None = None,
    status: str = "partial",
    descriptor_contract: Mapping[str, Any] | None = None,
    split_scientific_sha256: str | None = None,
    checkpoint_sha256: str | None = None,
    execution_profile_name: str = "frozen_deterministic_math",
    execution_profile_sha256: str | None = None,
    runtime_attestation_sha256: str | None = None,
    generation_commit: str | None = B2_COMMIT,
) -> dict[str, Any]:
    contract = descriptor_contract or subject.descriptor_contract(_config(), REPO_ROOT)
    payload = {
        "schema_version": 1,
        "status": status,
        "artifact_schema_version": 1,
        "split_scientific_hash_version": 2,
        "split_scientific_sha256": split_scientific_sha256 or SPLIT_V2,
        "checkpoint_sha256": checkpoint_sha256 or CHECKPOINT_SHA256,
        "execution_profile_name": execution_profile_name,
        "execution_profile_sha256": execution_profile_sha256 or PROFILE_SHA256,
        "runtime_attestation_sha256": runtime_attestation_sha256 or ("c" * 64),
        "descriptor_contract": dict(contract),
        "planned_stable_sample_ids": [row.stable_sample_id for row in plan],
        "samples": list(entries or []),
    }
    if generation_commit is not None:
        payload["generation_commit"] = generation_commit
    return payload


def _expected_run_provenance(
    *,
    split_scientific_sha256: str | None = None,
    checkpoint_sha256: str | None = None,
    execution_profile_name: str = "frozen_deterministic_math",
    execution_profile_sha256: str | None = None,
    runtime_attestation_sha256: str | None = None,
    generation_commit: str | None = B2_COMMIT,
) -> dict[str, Any]:
    payload = {
        "split_scientific_sha256": split_scientific_sha256 or SPLIT_V2,
        "checkpoint_sha256": checkpoint_sha256 or CHECKPOINT_SHA256,
        "execution_profile_name": execution_profile_name,
        "execution_profile_sha256": execution_profile_sha256 or PROFILE_SHA256,
        "runtime_attestation_sha256": runtime_attestation_sha256 or ("c" * 64),
    }
    if generation_commit is not None:
        payload["generation_commit"] = generation_commit
    return payload


def _resume_call_kwargs(
    *,
    plan: tuple[subject.PlannedSample, ...],
    accepted_config: subject.TeacherCacheConfig | None = None,
    descriptor_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    split_hash = (
        accepted_config.split_scientific_sha256
        if accepted_config is not None
        else SPLIT_V2
    )
    checkpoint_hash = (
        accepted_config.checkpoint_sha256
        if accepted_config is not None
        else CHECKPOINT_SHA256
    )
    return {
        "plan": plan,
        "expected_descriptor_contract": descriptor_contract
        or subject.descriptor_contract(_config(), REPO_ROOT),
        "expected_run_provenance": _expected_run_provenance(
            split_scientific_sha256=split_hash,
            checkpoint_sha256=checkpoint_hash,
        ),
    }


def _aligned_partial(
    *,
    plan: tuple[subject.PlannedSample, ...],
    accepted_config: subject.TeacherCacheConfig,
    entries: list[dict[str, Any]] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    return _partial_manifest_skeleton(
        plan=plan,
        entries=entries,
        split_scientific_sha256=accepted_config.split_scientific_sha256,
        checkpoint_sha256=accepted_config.checkpoint_sha256,
        **overrides,
    )


def test_option_a_pt_contains_only_scientific_record_and_scientific_hash(
    tmp_path: Path,
) -> None:
    record = _scientific_record_fixture()
    stable_id = record["stable_sample_id"]
    destination = _sample_pt_path(tmp_path / "run", stable_id)
    destination.parent.mkdir(parents=True)

    entry = subject.write_sample_atomic(destination, record)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="TypedStorage is deprecated",
            category=UserWarning,
        )
        payload = torch.load(destination, map_location="cpu", weights_only=True)
    assert set(payload) == {"scientific_record", "record_scientific_sha256"}
    assert "record_file_sha256" not in payload
    assert "record_file_sha256" not in payload["scientific_record"]
    assert payload["record_scientific_sha256"] == subject.record_scientific_sha256(record)
    assert payload["scientific_record"] == _scientific_only(record)
    assert entry.stable_sample_id == stable_id
    assert entry.relative_path == f"samples/{stable_id}.pt"
    assert entry.record_scientific_sha256 == payload["record_scientific_sha256"]
    assert entry.record_file_sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert entry.record_file_sha256 != entry.record_scientific_sha256


def test_record_file_sha256_lives_only_in_manifest_entry_after_persistence(
    tmp_path: Path,
    accepted_manifest: dict[str, Any],
    accepted_config: subject.TeacherCacheConfig,
) -> None:
    plan = subject.build_generation_plan(accepted_manifest, accepted_config)
    run_dir = tmp_path / "run"
    record = _scientific_record_fixture()
    record["stable_sample_id"] = plan[0].stable_sample_id
    destination = _sample_pt_path(run_dir, plan[0].stable_sample_id)
    destination.parent.mkdir(parents=True)
    entry = subject.write_sample_atomic(destination, record)
    partial = _partial_manifest_skeleton(
        plan=plan,
        entries=[
            {
                "stable_sample_id": entry.stable_sample_id,
                "relative_path": entry.relative_path,
                "record_scientific_sha256": entry.record_scientific_sha256,
                "record_file_sha256": entry.record_file_sha256,
            }
        ],
    )
    partial_path = run_dir / "partial_manifest.json"
    subject.write_partial_manifest_atomic(partial_path, partial)
    loaded = json.loads(partial_path.read_text(encoding="utf-8"))
    sample_entry = loaded["samples"][0]
    assert sample_entry["record_file_sha256"] == entry.record_file_sha256
    payload = torch.load(destination, map_location="cpu", weights_only=True)
    assert "record_file_sha256" not in payload
    final = subject.build_final_manifest(
        partial_manifest=loaded,
        entries=(entry,),
        plan=plan[:1],
    )
    assert final["status"] == "passed"
    assert final["samples"][0]["record_file_sha256"] == entry.record_file_sha256
    assert "record_file_sha256" not in torch.load(
        destination, map_location="cpu", weights_only=True
    )


def test_write_sample_atomic_refuses_overwrite(tmp_path: Path) -> None:
    record = _scientific_record_fixture()
    destination = _sample_pt_path(tmp_path / "run", record["stable_sample_id"])
    destination.parent.mkdir(parents=True)
    subject.write_sample_atomic(destination, record)
    before = destination.read_bytes()
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_SAMPLE_EXISTS"):
        subject.write_sample_atomic(destination, record)
    assert destination.read_bytes() == before


def test_write_sample_atomic_publishes_via_link_not_empty_excl_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _scientific_record_fixture()
    destination = _sample_pt_path(tmp_path / "run", record["stable_sample_id"])
    destination.parent.mkdir(parents=True)
    link_calls: list[tuple[Path, Path]] = []
    open_calls: list[tuple[Path, int]] = []
    real_link = os.link
    real_open = os.open

    def tracking_link(src: str | bytes | os.PathLike[str], dst: str | bytes | os.PathLike[str]) -> None:
        link_calls.append((Path(src), Path(dst)))
        real_link(src, dst)

    def tracking_open(
        path: str | bytes | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *args: Any,
        **kwargs: Any,
    ) -> int:
        open_calls.append((Path(path), int(flags)))
        return real_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(os, "link", tracking_link)
    monkeypatch.setattr(os, "open", tracking_open)
    subject.write_sample_atomic(destination, record)
    assert link_calls, "destination must be published via os.link"
    assert link_calls[0][1].resolve() == destination.resolve()
    assert link_calls[0][0].parent.resolve() == destination.parent.resolve()
    destination_excl_claims = [
        flags
        for path, flags in open_calls
        if path.resolve() == destination.resolve()
        and (flags & os.O_CREAT)
        and (flags & os.O_EXCL)
    ]
    assert destination_excl_claims == [], "must not create an empty O_EXCL destination claim"


def test_write_sample_atomic_crash_before_publish_leaves_no_planned_empty_pt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _scientific_record_fixture()
    run_dir = tmp_path / "run"
    samples = run_dir / "samples"
    samples.mkdir(parents=True)
    destination = _sample_pt_path(run_dir, record["stable_sample_id"])

    def boom_link(
        _src: str | bytes | os.PathLike[str],
        _dst: str | bytes | os.PathLike[str],
    ) -> None:
        raise OSError("simulated crash before link publish")

    monkeypatch.setattr(os, "link", boom_link)
    with pytest.raises(OSError, match="simulated crash before link publish"):
        subject.write_sample_atomic(destination, record)
    assert not destination.exists()
    assert list(samples.iterdir()) == []

    monkeypatch.undo()
    entry = subject.write_sample_atomic(destination, record)
    assert destination.is_file()
    assert destination.stat().st_size > 0
    assert entry.record_file_sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()


def test_claim_new_run_directory_refuses_existing(tmp_path: Path) -> None:
    run_dir = tmp_path / "existing"
    run_dir.mkdir()
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_RUN_EXISTS"):
        subject.claim_new_run_directory(run_dir)
    fresh = tmp_path / "fresh"
    subject.claim_new_run_directory(fresh)
    assert not fresh.exists()


def test_interrupted_temp_write_leaves_no_destination_or_passed_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_manifest: dict[str, Any],
    accepted_config: subject.TeacherCacheConfig,
) -> None:
    plan = subject.build_generation_plan(accepted_manifest, accepted_config)
    run_dir = tmp_path / "run"
    samples = run_dir / "samples"
    samples.mkdir(parents=True)
    record = _scientific_record_fixture()
    record["stable_sample_id"] = plan[0].stable_sample_id
    destination = _sample_pt_path(run_dir, plan[0].stable_sample_id)

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("simulated interrupted torch.save")

    monkeypatch.setattr(torch, "save", boom)
    with pytest.raises(OSError, match="simulated interrupted torch.save"):
        subject.write_sample_atomic(destination, record)
    assert not destination.exists()
    leftovers = list(samples.glob("*"))
    assert leftovers == []

    partial = _aligned_partial(plan=plan, accepted_config=accepted_config, status="passed")
    with pytest.raises(
        subject.TeacherCacheError, match="B2_CACHE_PARTIAL_STATUS_INVALID"
    ):
        subject.write_partial_manifest_atomic(run_dir / "partial_manifest.json", partial)
    assert not (run_dir / "partial_manifest.json").exists()
    assert not (run_dir / "final_manifest.json").exists()


def test_partial_manifest_forbidden_from_status_passed(tmp_path: Path) -> None:
    plan_ids = ["a" * 64, "b" * 64]
    fake_plan = tuple(
        subject.PlannedSample(
            stable_sample_id=sample_id,
            membership="training",
            category="bottle",
            image_label=0,
            anomaly_type="good",
            image_identity="bottle/test/good/000.png",
            mask_identity=None,
        )
        for sample_id in plan_ids
    )
    partial = _partial_manifest_skeleton(plan=fake_plan, status="passed")
    with pytest.raises(
        subject.TeacherCacheError, match="B2_CACHE_PARTIAL_STATUS_INVALID"
    ):
        subject.write_partial_manifest_atomic(tmp_path / "partial_manifest.json", partial)


def test_validate_resume_state_requires_valid_partial_manifest(
    tmp_path: Path,
    accepted_manifest: dict[str, Any],
    accepted_config: subject.TeacherCacheConfig,
) -> None:
    plan = subject.build_generation_plan(accepted_manifest, accepted_config)
    run_dir = tmp_path / "run"
    (run_dir / "samples").mkdir(parents=True)
    kwargs = _resume_call_kwargs(plan=plan, accepted_config=accepted_config)
    with pytest.raises(
        subject.TeacherCacheError, match="B2_CACHE_RESUME_MANIFEST_INVALID"
    ):
        subject.validate_resume_state(run_dir, {}, **kwargs)
    with pytest.raises(
        subject.TeacherCacheError, match="B2_CACHE_RESUME_MANIFEST_INVALID"
    ):
        subject.validate_resume_state(
            run_dir,
            _aligned_partial(
                plan=plan, accepted_config=accepted_config, status="passed"
            ),
            **kwargs,
        )


def test_validate_resume_state_rejects_hash_and_contract_drift(
    tmp_path: Path,
    accepted_manifest: dict[str, Any],
    accepted_config: subject.TeacherCacheConfig,
) -> None:
    plan = subject.build_generation_plan(accepted_manifest, accepted_config)
    run_dir = tmp_path / "run"
    (run_dir / "samples").mkdir(parents=True)
    record = _scientific_record_fixture()
    record["stable_sample_id"] = plan[0].stable_sample_id
    destination = _sample_pt_path(run_dir, plan[0].stable_sample_id)
    entry = subject.write_sample_atomic(destination, record)
    kwargs = _resume_call_kwargs(plan=plan, accepted_config=accepted_config)
    base_entry = {
        "stable_sample_id": entry.stable_sample_id,
        "relative_path": entry.relative_path,
        "record_scientific_sha256": entry.record_scientific_sha256,
        "record_file_sha256": entry.record_file_sha256,
    }
    wrong_scientific = _aligned_partial(
        plan=plan,
        accepted_config=accepted_config,
        entries=[{**base_entry, "record_scientific_sha256": "0" * 64}],
    )
    with pytest.raises(
        subject.TeacherCacheError, match="B2_CACHE_RESUME_SCIENTIFIC_HASH_MISMATCH"
    ):
        subject.validate_resume_state(run_dir, wrong_scientific, **kwargs)
    wrong_file = _aligned_partial(
        plan=plan,
        accepted_config=accepted_config,
        entries=[{**base_entry, "record_file_sha256": "1" * 64}],
    )
    with pytest.raises(
        subject.TeacherCacheError, match="B2_CACHE_RESUME_FILE_HASH_MISMATCH"
    ):
        subject.validate_resume_state(run_dir, wrong_file, **kwargs)
    drifted_contract = dict(kwargs["expected_descriptor_contract"])
    drifted_contract["descriptor_implementation_sha256"] = "0" * 64
    drifted = _aligned_partial(
        plan=plan,
        accepted_config=accepted_config,
        entries=[base_entry],
        descriptor_contract=drifted_contract,
    )
    with pytest.raises(
        subject.TeacherCacheError, match="B2_CACHE_RESUME_PROVENANCE_DRIFT"
    ):
        subject.validate_resume_state(run_dir, drifted, **kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("split_scientific_sha256", "0" * 64),
        ("checkpoint_sha256", "1" * 64),
        ("execution_profile_name", "wrong_profile"),
        ("execution_profile_sha256", "2" * 64),
        ("runtime_attestation_sha256", "3" * 64),
        ("generation_commit", "9" * 40),
    ],
)
def test_validate_resume_state_rejects_outer_run_provenance_value_drift(
    tmp_path: Path,
    accepted_manifest: dict[str, Any],
    accepted_config: subject.TeacherCacheConfig,
    field: str,
    value: str,
) -> None:
    plan = subject.build_generation_plan(accepted_manifest, accepted_config)
    run_dir = tmp_path / "run"
    (run_dir / "samples").mkdir(parents=True)
    record = _scientific_record_fixture()
    record["stable_sample_id"] = plan[0].stable_sample_id
    entry = subject.write_sample_atomic(
        _sample_pt_path(run_dir, plan[0].stable_sample_id), record
    )
    kwargs = _resume_call_kwargs(plan=plan, accepted_config=accepted_config)
    partial = _aligned_partial(
        plan=plan,
        accepted_config=accepted_config,
        entries=[
            {
                "stable_sample_id": entry.stable_sample_id,
                "relative_path": entry.relative_path,
                "record_scientific_sha256": entry.record_scientific_sha256,
                "record_file_sha256": entry.record_file_sha256,
            }
        ],
    )
    partial[field] = value
    with pytest.raises(
        subject.TeacherCacheError, match="B2_CACHE_RESUME_PROVENANCE_DRIFT"
    ):
        subject.validate_resume_state(run_dir, partial, **kwargs)


def test_validate_resume_state_rejects_orphan_files_before_reuse(
    tmp_path: Path,
    accepted_manifest: dict[str, Any],
    accepted_config: subject.TeacherCacheConfig,
) -> None:
    plan = subject.build_generation_plan(accepted_manifest, accepted_config)
    run_dir = tmp_path / "run"
    samples = run_dir / "samples"
    samples.mkdir(parents=True)
    record = _scientific_record_fixture()
    record["stable_sample_id"] = plan[0].stable_sample_id
    entry = subject.write_sample_atomic(
        _sample_pt_path(run_dir, plan[0].stable_sample_id), record
    )
    kwargs = _resume_call_kwargs(plan=plan, accepted_config=accepted_config)
    partial = _aligned_partial(
        plan=plan,
        accepted_config=accepted_config,
        entries=[
            {
                "stable_sample_id": entry.stable_sample_id,
                "relative_path": entry.relative_path,
                "record_scientific_sha256": entry.record_scientific_sha256,
                "record_file_sha256": entry.record_file_sha256,
            }
        ],
    )
    (samples / "orphan.pt").write_bytes(b"orphan")
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_ORPHAN_ARTIFACT"):
        subject.validate_resume_state(run_dir, partial, **kwargs)
    (samples / "orphan.pt").unlink()
    (samples / f".{plan[0].stable_sample_id}.pt.tmp").write_bytes(b"tmp")
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_ORPHAN_ARTIFACT"):
        subject.validate_resume_state(run_dir, partial, **kwargs)


def test_validate_resume_state_recomputes_hashes_before_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_manifest: dict[str, Any],
    accepted_config: subject.TeacherCacheConfig,
) -> None:
    plan = subject.build_generation_plan(accepted_manifest, accepted_config)
    run_dir = tmp_path / "run"
    (run_dir / "samples").mkdir(parents=True)
    record = _scientific_record_fixture()
    record["stable_sample_id"] = plan[0].stable_sample_id
    destination = _sample_pt_path(run_dir, plan[0].stable_sample_id)
    entry = subject.write_sample_atomic(destination, record)
    kwargs = _resume_call_kwargs(plan=plan, accepted_config=accepted_config)
    partial = _aligned_partial(
        plan=plan,
        accepted_config=accepted_config,
        entries=[
            {
                "stable_sample_id": entry.stable_sample_id,
                "relative_path": entry.relative_path,
                "record_scientific_sha256": entry.record_scientific_sha256,
                "record_file_sha256": entry.record_file_sha256,
            }
        ],
    )
    file_hash_calls: list[Path] = []
    scientific_hash_calls: list[Any] = []
    real_file_sha256 = subject._sha256_file
    real_scientific = subject.record_scientific_sha256

    def tracking_file(path: Path) -> str:
        file_hash_calls.append(Path(path))
        return real_file_sha256(path)

    def tracking_scientific(payload: Mapping[str, Any]) -> str:
        scientific_hash_calls.append(payload)
        return real_scientific(payload)

    monkeypatch.setattr(subject, "_sha256_file", tracking_file)
    monkeypatch.setattr(subject, "record_scientific_sha256", tracking_scientific)
    reused = subject.validate_resume_state(run_dir, partial, **kwargs)
    assert reused == (entry,)
    assert destination.resolve() in {path.resolve() for path in file_hash_calls}
    assert scientific_hash_calls


def test_audit_complete_coverage_requires_exact_plan_entry_file_sets(
    tmp_path: Path,
    accepted_manifest: dict[str, Any],
    accepted_config: subject.TeacherCacheConfig,
) -> None:
    plan = subject.build_generation_plan(accepted_manifest, accepted_config)
    run_dir = tmp_path / "run"
    samples = run_dir / "samples"
    samples.mkdir(parents=True)
    entries: list[subject.PersistedSampleEntry] = []
    for row in plan:
        record = _scientific_record_fixture()
        record["stable_sample_id"] = row.stable_sample_id
        record["membership"] = row.membership
        record["category"] = row.category
        record["image_label"] = row.image_label
        record["anomaly_type"] = row.anomaly_type
        record["image_identity"] = row.image_identity
        record["mask_identity"] = row.mask_identity
        entry = subject.write_sample_atomic(
            _sample_pt_path(run_dir, row.stable_sample_id), record
        )
        entries.append(entry)
    subject.audit_complete_coverage(run_dir, plan, tuple(entries))

    missing_entries = tuple(entries[:-1])
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_COVERAGE_MISMATCH"):
        subject.audit_complete_coverage(run_dir, plan, missing_entries)

    orphan = samples / "orphan.pt"
    orphan.write_bytes(b"orphan")
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_ORPHAN_ARTIFACT"):
        subject.audit_complete_coverage(run_dir, plan, tuple(entries))
    orphan.unlink()

    sidecar = samples / f"{plan[0].stable_sample_id}.pt.sha256"
    sidecar.write_text("sidecar\n", encoding="utf-8")
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_ORPHAN_ARTIFACT"):
        subject.audit_complete_coverage(run_dir, plan, tuple(entries))
    sidecar.unlink()

    temp = samples / f".{plan[0].stable_sample_id}.pt.tmp"
    temp.write_bytes(b"temp")
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_ORPHAN_ARTIFACT"):
        subject.audit_complete_coverage(run_dir, plan, tuple(entries))
    temp.unlink()

    lock = samples / ".samples.lock"
    lock.write_text("lock\n", encoding="utf-8")
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_ORPHAN_ARTIFACT"):
        subject.audit_complete_coverage(run_dir, plan, tuple(entries))
    lock.unlink()

    duplicate_entries = tuple(entries) + (entries[0],)
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_COVERAGE_MISMATCH"):
        subject.audit_complete_coverage(run_dir, plan, duplicate_entries)

    remapped = [
        subject.PersistedSampleEntry(
            stable_sample_id=entry.stable_sample_id,
            relative_path=(
                entry.relative_path
                if entry is not entries[0]
                else f"samples/{plan[1].stable_sample_id}.pt"
            ),
            record_scientific_sha256=entry.record_scientific_sha256,
            record_file_sha256=entry.record_file_sha256,
        )
        for entry in entries
    ]
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_COVERAGE_MISMATCH"):
        subject.audit_complete_coverage(run_dir, plan, tuple(remapped))

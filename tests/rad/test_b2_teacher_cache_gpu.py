"""B2-02B production GPU teacher-cache fail-closed and persistence tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from rad.phase_b import b2_teacher_cache as subject
from tests.rad.b2_hermetic import (
    EXPECTED_CHECKPOINT_SHA256,
    write_b2_split_fixture,
    write_hermetic_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPO_ROOT / "tools" / "create_b2_teacher_cache.py"
LAUNCHER_PATH = REPO_ROOT / "tools" / "run_with_execution_profile.py"
PROFILE_PATH = REPO_ROOT / "configs" / "execution" / "frozen_deterministic_math.json"
CONFIG_PATH = REPO_ROOT / "configs" / "phase_b" / "b2_teacher_cache_gate_c.json"
# Live CUDA parity only: production artifacts are not required for portable CPU CI.
OFFICIAL_SPLIT = Path(
    "/root/autodl-tmp/AD-phase-b2-gate-c/artifacts/phase_b/b2_gate_c/"
    "b2c-20260721-v2-final-18bac04/split_manifest.json"
)
OFFICIAL_CHECKPOINT = Path(
    "/root/autodl-tmp/AD/runs/baseline/mvtec_to_visa/"
    "seed_111_official_bs8/checkpoints/epoch_2.pth"
)
EXPECTED_PROFILE_SHA256 = (
    "7af8dba39633743da0380fef9710940cded655f68c9efa8f84f5a52aeddb3c8d"
)
MVTEC_ROOT = Path("/root/autodl-tmp/data/mvtec")
CUDA_REQUIRED = not torch.cuda.is_available()


def _tensor_record() -> dict[str, Any]:
    from tests.rad.test_b2_teacher_cache import _scientific_record_fixture

    record = _scientific_record_fixture()
    tensor = torch.arange(20, dtype=torch.float32).reshape(1, 1, 4, 5)
    meta = record["tensors"]["causal_map:12:6"]
    meta["tensor"] = tensor
    meta["digest"] = subject.canonical_tensor_digest(
        "causal_map:12:6",
        tensor,
        ("batch", "channel", "height", "width"),
    )
    return record


def test_tensor_bearing_option_a_record_round_trips_with_digest_verification(
    tmp_path: Path,
) -> None:
    record = _tensor_record()
    stable_id = record["stable_sample_id"]
    entry = subject.write_sample_atomic(tmp_path / f"{stable_id}.pt", record)
    payload = torch.load(tmp_path / f"{stable_id}.pt", map_location="cpu", weights_only=True)
    loaded = payload["scientific_record"]
    assert set(payload) == {"scientific_record", "record_scientific_sha256"}
    assert "record_file_sha256" not in loaded
    assert torch.equal(
        loaded["tensors"]["causal_map:12:6"]["tensor"],
        record["tensors"]["causal_map:12:6"]["tensor"],
    )
    assert entry.record_scientific_sha256 == subject.record_scientific_sha256(loaded)


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda record: record["tensors"]["causal_map:12:6"].__setitem__(
                "tensor", torch.ones(1, 1, 4, 5, dtype=torch.float64)
            ),
            "B2_CACHE_TENSOR_DTYPE_INVALID",
        ),
        (
            lambda record: record["tensors"]["causal_map:12:6"].__setitem__(
                "tensor", torch.zeros(1, 1, 4, 5, dtype=torch.float32)
            ),
            "B2_CACHE_TENSOR_DIGEST_MISMATCH",
        ),
    ],
)
def test_tensor_bearing_option_a_record_fails_closed_on_invalid_values(
    tmp_path: Path,
    mutate: Any,
    code: str,
) -> None:
    record = _tensor_record()
    mutate(record)
    with pytest.raises(subject.TeacherCacheError, match=code):
        subject.write_sample_atomic(tmp_path / f"{record['stable_sample_id']}.pt", record)


def test_production_teacher_load_rejects_cpu_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnsafeBundle:
        model = torch.nn.Linear(1, 1)
        layer_transforms = torch.nn.ModuleDict()
        cross_attn = None
        features_list = [6, 12, 18, 24]
        image_size = 4
        device = torch.device("cpu")

    monkeypatch.setattr(subject, "load_teacher_bundle", lambda *_a, **_k: UnsafeBundle())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_CUDA_REQUIRED"):
        subject.ProductionTeacher.load(
            Path("/checkpoint.pth"),
            candidate_layers=(6, 12, 18, 24),
        )


def test_production_teacher_load_rejects_when_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_CUDA_REQUIRED"):
        subject.ProductionTeacher.load(
            Path("/checkpoint.pth"),
            candidate_layers=(6, 12, 18, 24),
        )


def test_production_teacher_forward_rejects_amp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher = subject.ProductionTeacher(
        bundle=SimpleNamespace(image_size=4),
        candidate_layers=(6, 12, 18, 24),
    )
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda: True)
    sample = subject.PlannedSample(
        stable_sample_id="a" * 64,
        membership="training",
        category="bottle",
        image_label=0,
        anomaly_type="good",
        image_identity="bottle/test/good/000.png",
        mask_identity=None,
    )
    contract = subject.CacheContract(
        candidate_layers=(6, 12, 18, 24),
        prediction_depths=(12, 18, 24),
        backbone_depth=24,
        expected_sample_ids=frozenset({sample.stable_sample_id}),
        map_shape=(1, 1, 4, 5),
        map_dimension_semantics=("batch", "channel", "height", "width"),
        production_mode=True,
    )
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_AMP_FORBIDDEN"):
        teacher.forward(
            sample=sample,
            image=torch.zeros(1, 3, 4, 4),
            anomalous_mask=torch.zeros(4, 4),
            contract=contract,
        )


def test_require_production_teacher_rejects_fixture_and_unknown() -> None:
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_TEST_TEACHER_FORBIDDEN"):
        subject.require_production_teacher(SimpleNamespace(artifact_kind="test_fixture"))
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_TEST_TEACHER_FORBIDDEN"):
        subject.require_production_teacher(SimpleNamespace(artifact_kind="other"))


def test_target_access_guard_rejects_visa_paths() -> None:
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_TARGET_ACCESS_FORBIDDEN"):
        subject.forbid_target_domain_access(Path("/root/autodl-tmp/data/Visa/candle"))
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_TARGET_ACCESS_FORBIDDEN"):
        subject.forbid_target_domain_access("dataset/visa/sample.png")


def test_incomplete_finalization_fails_closed(tmp_path: Path) -> None:
    config = subject.load_teacher_cache_config(CONFIG_PATH)
    manifest = json.loads(write_b2_split_fixture(tmp_path / "split.json").read_text(encoding="utf-8"))
    plan = subject.build_generation_plan(manifest, config)
    entries = [
        subject.PersistedSampleEntry(
            stable_sample_id=sample.stable_sample_id,
            relative_path=subject.sample_relative_path(sample.stable_sample_id),
            record_scientific_sha256="a" * 64,
            record_file_sha256="b" * 64,
        )
        for sample in plan[:31]
    ]
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    for entry in entries:
        (tmp_path / entry.relative_path).write_bytes(b"x")
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_COVERAGE_MISMATCH"):
        subject.audit_complete_coverage(tmp_path, plan, entries)


def test_unplanned_sample_rejected_by_contract() -> None:
    lattice = subject.expected_lattice((6, 12, 18, 24), (12, 18, 24))
    maps = {
        identity: torch.zeros(1, 1, 4, 5, dtype=torch.float32) for identity in lattice
    }
    output = subject.TeacherOutput(
        sample_id="c" * 64,
        image_label=0,
        anomalous_mask=None,
        maps=maps,
        map_dimension_semantics={identity: ("batch", "channel", "height", "width") for identity in maps},
        descriptor_source_identities=frozenset(maps),
        artifact_kind="production",
    )
    contract = subject.CacheContract(
        candidate_layers=(6, 12, 18, 24),
        prediction_depths=(12, 18, 24),
        backbone_depth=24,
        expected_sample_ids=frozenset({"a" * 64}),
        map_shape=(1, 1, 4, 5),
        map_dimension_semantics=("batch", "channel", "height", "width"),
        production_mode=True,
    )
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_UNEXPECTED_SAMPLE"):
        subject.validate_teacher_output(output, contract)


def test_missing_map_rejected_by_contract() -> None:
    lattice = subject.expected_lattice((6, 12, 18, 24), (12, 18, 24))
    missing = subject.MapIdentity(12, 6)
    maps = {
        identity: torch.zeros(1, 1, 4, 5, dtype=torch.float32)
        for identity in lattice
        if identity != missing
    }
    output = subject.TeacherOutput(
        sample_id="a" * 64,
        image_label=0,
        anomalous_mask=None,
        maps=maps,
        map_dimension_semantics={identity: ("batch", "channel", "height", "width") for identity in maps},
        descriptor_source_identities=frozenset(maps),
        artifact_kind="production",
    )
    contract = subject.CacheContract(
        candidate_layers=(6, 12, 18, 24),
        prediction_depths=(12, 18, 24),
        backbone_depth=24,
        expected_sample_ids=frozenset({"a" * 64}),
        map_shape=(1, 1, 4, 5),
        map_dimension_semantics=("batch", "channel", "height", "width"),
        production_mode=True,
    )
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_MAP_LATTICE_MISMATCH"):
        subject.validate_teacher_output(output, contract)


def test_non_float32_map_rejected_by_contract() -> None:
    lattice = subject.expected_lattice((6, 12, 18, 24), (12, 18, 24))
    maps = {
        identity: torch.zeros(1, 1, 4, 5, dtype=torch.float64) for identity in lattice
    }
    output = subject.TeacherOutput(
        sample_id="a" * 64,
        image_label=0,
        anomalous_mask=None,
        maps=maps,
        map_dimension_semantics={identity: ("batch", "channel", "height", "width") for identity in maps},
        descriptor_source_identities=frozenset(maps),
        artifact_kind="production",
    )
    contract = subject.CacheContract(
        candidate_layers=(6, 12, 18, 24),
        prediction_depths=(12, 18, 24),
        backbone_depth=24,
        expected_sample_ids=frozenset({"a" * 64}),
        map_shape=(1, 1, 4, 5),
        map_dimension_semantics=("batch", "channel", "height", "width"),
        production_mode=True,
    )
    with pytest.raises(subject.TeacherCacheError, match="B2_CACHE_TENSOR_DTYPE_INVALID"):
        subject.validate_teacher_output(output, contract)


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in (
        "RAD_EXECUTION_PROFILE_BOOTSTRAPPED",
        "RAD_EXECUTION_PROFILE_PATH",
        "RAD_EXECUTION_PROFILE_SHA256",
    ):
        env.pop(key, None)
    return env


def test_cli_outside_launcher_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "output_root"
    root.mkdir()
    run_dir = root / "run"
    split = write_b2_split_fixture(tmp_path / "split.json")
    checkpoint, _ = write_hermetic_checkpoint(tmp_path / "ckpt.pth")
    args = [
        sys.executable,
        str(CLI_PATH),
        "--config",
        str(CONFIG_PATH),
        "--seed",
        "111",
        "--output-dir",
        str(run_dir),
        "--output-root",
        str(root),
        "--split-manifest",
        str(split),
        "--checkpoint",
        str(checkpoint),
        "--expected-checkpoint-sha256",
        EXPECTED_CHECKPOINT_SHA256,
    ]
    proc = subprocess.run(
        args,
        cwd=REPO_ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode != 0
    assert "B2_CACHE_BOOTSTRAP_REQUIRED" in proc.stdout + proc.stderr
    assert not (root / "manifest.json").exists()


def test_cli_wrong_checkpoint_hash_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "output_root"
    root.mkdir()
    run_dir = root / "run"
    split = write_b2_split_fixture(tmp_path / "split.json")
    checkpoint, _ = write_hermetic_checkpoint(tmp_path / "ckpt.pth")
    harness = """
import runpy, subprocess, sys
from dataclasses import replace
from pathlib import Path
from rad.runtime.execution_profile import apply_execution_profile
import rad.phase_b.b2_teacher_cache as cache_mod
from tests.rad import b2_hermetic as hermetic
attestation = apply_execution_profile()
module = runpy.run_path(sys.argv[1], run_name='cli')
module['main'].__globals__['_apply_profile'] = lambda _repo: attestation
ckpt = Path(sys.argv[2]).resolve()
real_load = cache_mod.load_teacher_cache_config
cache_mod.load_teacher_cache_config = lambda path: replace(real_load(path), checkpoint_path=ckpt)
head = subprocess.check_output(['git','-C',str(Path(sys.argv[1]).resolve().parents[1]),'rev-parse','HEAD'], text=True).strip()
module['main'].__globals__['_derive_repository_identity'] = (
    lambda repo, *, require_clean=True: hermetic.synthetic_teacher_cache_identity(
        head_commit=head, worktree_clean=True, head_is_descendant=True
    )
)
raise SystemExit(module['main'](sys.argv[3:]))
"""
    proc = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER_PATH),
            "--profile",
            str(PROFILE_PATH),
            "--expected-sha256",
            EXPECTED_PROFILE_SHA256,
            "--",
            sys.executable,
            "-c",
            harness,
            str(CLI_PATH),
            str(checkpoint),
            "--config",
            str(CONFIG_PATH),
            "--seed",
            "111",
            "--output-dir",
            str(run_dir),
            "--output-root",
            str(root),
            "--split-manifest",
            str(split),
            "--checkpoint",
            str(checkpoint),
            "--expected-checkpoint-sha256",
            "0" * 64,
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode != 0
    assert "B2_CACHE_CHECKPOINT_HASH_MISMATCH" in proc.stdout + proc.stderr


def test_cli_wrong_split_v2_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "output_root"
    root.mkdir()
    run_dir = root / "run"
    split = tmp_path / "split.json"
    payload = json.loads(write_b2_split_fixture(tmp_path / "base_split.json").read_text(encoding="utf-8"))
    payload["scientific_hash_contract"]["canonical_scientific_hash_v2"] = "a" * 64
    split.write_text(json.dumps(payload), encoding="utf-8")
    checkpoint, _ = write_hermetic_checkpoint(tmp_path / "ckpt.pth")
    harness = """
import runpy, subprocess, sys
from dataclasses import replace
from pathlib import Path
from rad.runtime.execution_profile import apply_execution_profile
import rad.phase_b.b2_teacher_cache as cache_mod
from tests.rad import b2_hermetic as hermetic
attestation = apply_execution_profile()
module = runpy.run_path(sys.argv[1], run_name='cli')
module['main'].__globals__['_apply_profile'] = lambda _repo: attestation
ckpt = Path(sys.argv[2]).resolve()
real_load = cache_mod.load_teacher_cache_config
real_validate = cache_mod.validate_checkpoint_bytes
cache_mod.load_teacher_cache_config = lambda path: replace(real_load(path), checkpoint_path=ckpt)
def hermetic_validate(path, expected):
    if Path(path).resolve() != ckpt:
        return real_validate(path, expected)
    if expected != hermetic.EXPECTED_CHECKPOINT_SHA256:
        raise cache_mod.TeacherCacheError('B2_CACHE_CHECKPOINT_HASH_MISMATCH', 'hash')
    return expected
cache_mod.validate_checkpoint_bytes = hermetic_validate
head = subprocess.check_output(['git','-C',str(Path(sys.argv[1]).resolve().parents[1]),'rev-parse','HEAD'], text=True).strip()
module['main'].__globals__['_derive_repository_identity'] = (
    lambda repo, *, require_clean=True: hermetic.synthetic_teacher_cache_identity(
        head_commit=head, worktree_clean=True, head_is_descendant=True
    )
)
raise SystemExit(module['main'](sys.argv[3:]))
"""
    proc = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER_PATH),
            "--profile",
            str(PROFILE_PATH),
            "--expected-sha256",
            EXPECTED_PROFILE_SHA256,
            "--",
            sys.executable,
            "-c",
            harness,
            str(CLI_PATH),
            str(checkpoint),
            "--config",
            str(CONFIG_PATH),
            "--seed",
            "111",
            "--output-dir",
            str(run_dir),
            "--output-root",
            str(root),
            "--split-manifest",
            str(split),
            "--checkpoint",
            str(checkpoint),
            "--expected-checkpoint-sha256",
            EXPECTED_CHECKPOINT_SHA256,
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode != 0
    assert "B2_CACHE_SPLIT_HASH_MISMATCH" in proc.stdout + proc.stderr


@pytest.mark.skipif(CUDA_REQUIRED, reason="CUDA required for live production parity")
def test_live_vs_persisted_descriptor_parity_one_normal_one_anomalous(
    tmp_path: Path,
) -> None:
    config = subject.load_teacher_cache_config(CONFIG_PATH)
    manifest = json.loads(OFFICIAL_SPLIT.read_text(encoding="utf-8"))
    plan = subject.build_generation_plan(manifest, config)
    normal = next(sample for sample in plan if sample.image_label == 0)
    anomalous = next(sample for sample in plan if sample.image_label == 1)
    teacher = subject.ProductionTeacher.load(
        OFFICIAL_CHECKPOINT,
        candidate_layers=config.candidate_layers,
    )
    subject.require_production_teacher(teacher)
    from rad.data.adapters.mvtec import MVTecAdapter
    from rad.data.adapters.preprocess import build_preprocess, preprocess_image, preprocess_mask

    adapter = MVTecAdapter(MVTEC_ROOT)
    records = {
        record.image_path.relative_to(MVTEC_ROOT).as_posix(): record
        for record in adapter.records(
            "test",
            categories=sorted({normal.category, anomalous.category}),
        )
    }
    preprocess = build_preprocess("ViT-L/14@336px", teacher.bundle.image_size)
    descriptor = subject.descriptor_contract(config, REPO_ROOT)

    for sample in (normal, anomalous):
        record = records[sample.image_identity]
        image = preprocess_image(adapter.open_image(record), preprocess).unsqueeze(0).cuda()
        raw_mask = preprocess_mask(adapter.open_mask(record), teacher.bundle.image_size)
        output = teacher.forward(
            sample=sample,
            image=image,
            anomalous_mask=raw_mask,
            contract=subject.CacheContract(
                candidate_layers=config.candidate_layers,
                prediction_depths=config.prediction_depths,
                backbone_depth=max(config.candidate_layers),
                expected_sample_ids=frozenset({sample.stable_sample_id}),
                map_shape=(1, 1, teacher.bundle.image_size, teacher.bundle.image_size),
                map_dimension_semantics=("batch", "channel", "height", "width"),
                production_mode=True,
            ),
        )
        assert len(output.maps) == 9
        assert all(tensor.dtype == torch.float32 for tensor in output.maps.values())
        first = next(iter(output.maps.values()))
        validated = subject.validate_teacher_output(
            output,
            subject.CacheContract(
                candidate_layers=config.candidate_layers,
                prediction_depths=config.prediction_depths,
                backbone_depth=max(config.candidate_layers),
                expected_sample_ids=frozenset({sample.stable_sample_id}),
                map_shape=tuple(first.shape),
                map_dimension_semantics=("batch", "channel", "height", "width"),
                production_mode=True,
            ),
        )
        live = subject.reconstruct_descriptors(validated)
        cumulative = subject.build_cumulative_maps(validated)
        assert set(cumulative) == {12, 18, 24}
        score = subject.compute_final_image_score(cumulative[24])
        assert score.dtype == torch.float32
        scientific = subject.build_scientific_record(
            sample=sample,
            validated=validated,
            cumulative=cumulative,
            image_score=score,
            config=config,
            descriptor=descriptor,
        )
        path = tmp_path / f"{sample.stable_sample_id}.pt"
        subject.write_sample_atomic(path, scientific)
        loaded = torch.load(path, map_location="cpu", weights_only=True)["scientific_record"]
        cached = subject.reconstruct_persisted_descriptors(loaded)
        for depth in (12, 18, 24):
            assert live[depth].dtype == cached[depth].dtype
            assert live[depth].shape == cached[depth].shape
            assert torch.equal(live[depth], cached[depth])

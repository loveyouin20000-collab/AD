from __future__ import annotations

import json
from pathlib import Path

import pytest

from rad.checkpoints.manifest_v1 import (
    SCHEMA_VERSION,
    CheckpointManifestV1,
    sha256_file,
    write_checkpoint_with_manifest,
    load_manifest,
    validate_manifest_against_file,
)


def test_write_and_validate_fusion_manifest(tmp_path: Path) -> None:
    ckpt = tmp_path / "best_gate_passed.pt"
    ckpt.write_bytes(b"fusion-bytes")
    digest = sha256_file(ckpt)
    manifest = CheckpointManifestV1(
        schema_version=SCHEMA_VERSION,
        stage="fusion",
        status="passed",
        checkpoint_sha256=digest,
        candidate_layers=(6, 12, 18, 24),
        source_dataset="mvtec",
        split_manifest_hash="split-111",
        preprocessing_hash="pre-v1",
        teacher_checkpoint_hash="teacher-v1",
        descriptor_stats_hash="stats-v1",
        upstream_fusion_checkpoint_hash=None,
        gates={"staged_training": True, "source_only_selection": True},
        reference_full_depth_metrics={
            "pixel_ap": 0.80,
            "pro": 0.90,
            "mean_sample_error": 0.20,
        },
    )
    write_checkpoint_with_manifest(ckpt, manifest)
    loaded = load_manifest(ckpt.with_suffix(".manifest.json"))
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.stage == "fusion"
    validate_manifest_against_file(ckpt, loaded)


def test_validate_rejects_hash_mismatch(tmp_path: Path) -> None:
    ckpt = tmp_path / "best_gate_passed.pt"
    ckpt.write_bytes(b"fusion-bytes")
    manifest = CheckpointManifestV1(
        schema_version=SCHEMA_VERSION,
        stage="fusion",
        status="passed",
        checkpoint_sha256="deadbeef",
        candidate_layers=(6, 12, 18, 24),
        source_dataset="mvtec",
        split_manifest_hash="split-111",
        preprocessing_hash="pre-v1",
        teacher_checkpoint_hash="teacher-v1",
        descriptor_stats_hash="stats-v1",
        upstream_fusion_checkpoint_hash=None,
        gates={"staged_training": True, "source_only_selection": True},
        reference_full_depth_metrics={
            "pixel_ap": 0.80,
            "pro": 0.90,
            "mean_sample_error": 0.20,
        },
    )
    side = ckpt.with_suffix(".manifest.json")
    side.write_text(json.dumps(manifest.to_dict()) + "\n")
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        validate_manifest_against_file(ckpt, load_manifest(side))

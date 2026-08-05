"""B2-06C LSE prerequisite materialization tests."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from rad.data.cache_dataset import TeacherCacheDataset
from rad.phase_b import b2_lse_prerequisites as prereq


def _record(sample_id: str, split: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "label": 0,
        "mask_path": "",
        "category": "bottle",
        "split": split,
        "maps": {
            6: {6: torch.zeros(2, 2)},
            12: {6: torch.zeros(2, 2), 12: torch.zeros(2, 2)},
            18: {6: torch.zeros(2, 2), 12: torch.zeros(2, 2), 18: torch.zeros(2, 2)},
            24: {
                6: torch.zeros(2, 2),
                12: torch.zeros(2, 2),
                18: torch.zeros(2, 2),
                24: torch.zeros(2, 2),
            },
        },
        "ingredients": {},
        "teacher_logits": torch.zeros(1),
        "candidate_layers": [6, 12, 18, 24],
        "preprocessing_hash": "prep",
        "split_hash": "split",
        "checkpoint_hash": "ckpt",
        "schema_version": 1,
    }


def _write_b2_cache(root: Path) -> None:
    samples = []
    for sid, split in [("train-1", "training"), ("cal-1", "calibration"), ("eval-1", "evaluation")]:
        rel = f"samples/{sid}.pt"
        (root / "samples").mkdir(parents=True, exist_ok=True)
        torch.save(_record(sid, split), root / rel)
        samples.append({"stable_sample_id": sid, "relative_path": rel})
    manifest = {
        "schema_version": 1,
        "cache_scientific_sha256": "cache",
        "candidate_layers": [6, 12, 18, 24],
        "checkpoint_sha256": "ckpt",
        "split_scientific_sha256": "split",
        "samples": samples,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def test_convert_b2_cache_writes_lse_train_and_calibration_datasets(tmp_path: Path) -> None:
    source = tmp_path / "b2_cache"
    _write_b2_cache(source)
    train = tmp_path / "out" / "train"
    cal = tmp_path / "out" / "calibration"

    receipt = prereq.convert_b2_cache_for_lse(
        source_cache=source,
        train_cache=train,
        calibration_cache=cal,
        data_path=tmp_path / "mvtec",
    )

    assert receipt["training_count"] == 1
    assert receipt["calibration_count"] == 1
    assert receipt["evaluation_ignored_count"] == 1
    assert TeacherCacheDataset(train).meta["split"] == "train"
    assert TeacherCacheDataset(cal).meta["split"] == "calibration"
    assert TeacherCacheDataset(train)[0]["sample_id"] == "train-1"
    assert TeacherCacheDataset(cal)[0]["sample_id"] == "cal-1"


def test_missing_b2_cache_manifest_fails_closed(tmp_path: Path) -> None:
    try:
        prereq.convert_b2_cache_for_lse(
            source_cache=tmp_path / "missing",
            train_cache=tmp_path / "train",
            calibration_cache=tmp_path / "calibration",
            data_path=tmp_path / "mvtec",
        )
    except prereq.B2LSEPrerequisiteError as exc:
        assert exc.code == "B2_LSE_PREREQ_SOURCE_CACHE_MANIFEST_MISSING"
    else:
        raise AssertionError("expected fail-closed missing manifest")

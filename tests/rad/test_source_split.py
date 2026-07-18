from __future__ import annotations

import json
from pathlib import Path

import pytest

from rad.data.split import (
    SampleRecord,
    build_source_split,
    load_samples_from_meta,
    write_manifest,
)


def _make_samples() -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for category in ("bottle", "cable"):
        for label in (0, 1):
            for idx in range(10):
                img = f"{category}/train/{'good' if label == 0 else 'defect'}/{idx:03d}.png"
                records.append(
                    SampleRecord(
                        sample_id=img,
                        image_path=img,
                        mask_path=(
                            ""
                            if label == 0
                            else f"{category}/ground_truth/defect/{idx:03d}.png"
                        ),
                        category=category,
                        label=label,
                    )
                )
    return records


def test_same_seed_produces_identical_split():
    samples = _make_samples()
    a = build_source_split(samples, calibration_fraction=0.2, seed=111)
    b = build_source_split(samples, calibration_fraction=0.2, seed=111)
    assert a == b


def test_train_and_calibration_have_no_overlap_and_cover_all():
    samples = _make_samples()
    rows = build_source_split(samples, calibration_fraction=0.2, seed=111)
    train_ids = {r["sample_id"] for r in rows if r["split"] == "train"}
    cal_ids = {r["sample_id"] for r in rows if r["split"] == "calibration"}
    all_ids = {s.sample_id for s in samples}
    assert train_ids.isdisjoint(cal_ids)
    assert train_ids | cal_ids == all_ids
    assert all(r["split"] in {"train", "calibration"} for r in rows)


def test_stratified_by_category_and_label():
    samples = _make_samples()
    rows = build_source_split(samples, calibration_fraction=0.2, seed=111)
    for category in ("bottle", "cable"):
        for label in (0, 1):
            stratum = [r for r in rows if r["category"] == category and r["label"] == label]
            n_cal = sum(1 for r in stratum if r["split"] == "calibration")
            # n=10, fraction=0.2 -> round(2)=2, and n_cal = min(9, 2) = 2
            assert n_cal == 2
            assert len(stratum) == 10


def test_singleton_stratum_stays_in_train():
    samples = [
        SampleRecord(
            sample_id="only/a.png",
            image_path="only/a.png",
            mask_path="",
            category="only",
            label=0,
        )
    ]
    rows = build_source_split(samples, calibration_fraction=0.2, seed=111)
    assert rows[0]["split"] == "train"


def test_write_manifest_refuses_overwrite_without_force(tmp_path: Path):
    path = tmp_path / "split.jsonl"
    rows = [{"sample_id": "a", "split": "train"}]
    write_manifest(path, rows, force=False)
    with pytest.raises(FileExistsError):
        write_manifest(path, rows, force=False)
    write_manifest(path, [{"sample_id": "b", "split": "calibration"}], force=True)
    loaded = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert loaded == [{"sample_id": "b", "split": "calibration"}]


def test_load_samples_from_meta(tmp_path: Path):
    meta = {
        "train": {
            "bottle": [
                {
                    "img_path": "bottle/train/good/000.png",
                    "mask_path": "",
                    "cls_name": "bottle",
                    "specie_name": "good",
                    "anomaly": 0,
                }
            ]
        },
        "test": {},
    }
    root = tmp_path / "mvtec"
    root.mkdir()
    (root / "meta.json").write_text(json.dumps(meta))
    samples = load_samples_from_meta(root, mode="train")
    assert len(samples) == 1
    assert samples[0].category == "bottle"
    assert samples[0].label == 0
    assert samples[0].sample_id == "bottle/train/good/000.png"

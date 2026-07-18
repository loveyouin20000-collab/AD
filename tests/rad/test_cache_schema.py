from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from rad.data.cache_schema import (
    SCHEMA_VERSION,
    CacheManifestError,
    build_sample_record,
    compute_file_hash,
    compute_preprocessing_hash,
    expected_map_layers,
    validate_sample_record,
    verify_shard,
    write_parquet_index,
    write_shard,
)


def _toy_maps(candidate_layers: tuple[int, ...], h: int = 4) -> dict[int, dict[int, torch.Tensor]]:
    maps: dict[int, dict[int, torch.Tensor]] = {}
    for depth in candidate_layers:
        maps[depth] = {
            layer: torch.randn(h, h)
            for layer in candidate_layers
            if layer <= depth
        }
    return maps


def _toy_ingredients(candidate_layers: tuple[int, ...]) -> dict:
    return {
        "patch_tokens": {layer: torch.randn(9, 8) for layer in candidate_layers},
        "anomaly_tokens": {depth: torch.randn(8) for depth in candidate_layers},
        "normal_tokens": {depth: torch.randn(8) for depth in candidate_layers},
    }


def test_expected_map_layers_are_causal_and_config_driven():
    assert expected_map_layers((6, 12, 18, 24)) == {
        6: (6,),
        12: (6, 12),
        18: (6, 12, 18),
        24: (6, 12, 18, 24),
    }
    assert expected_map_layers((8, 16)) == {8: (8,), 16: (8, 16)}


def test_validate_rejects_incomplete_checkpoint_maps():
    layers = (6, 12, 18, 24)
    maps = _toy_maps(layers)
    del maps[12][6]
    record = build_sample_record(
        sample_id="a.png",
        label=0,
        mask_path="",
        category="bottle",
        split="train",
        maps=maps,
        ingredients=_toy_ingredients(layers),
        teacher_logits=torch.randn(4, 4),
        candidate_layers=layers,
        preprocessing_hash="p" * 64,
        split_hash="s" * 64,
        checkpoint_hash="c" * 64,
    )
    with pytest.raises(CacheManifestError, match="incomplete"):
        validate_sample_record(record, candidate_layers=layers)


def test_validate_rejects_stale_hashes():
    layers = (6, 12)
    record = build_sample_record(
        sample_id="a.png",
        label=0,
        mask_path="",
        category="bottle",
        split="train",
        maps=_toy_maps(layers),
        ingredients=_toy_ingredients(layers),
        teacher_logits=torch.randn(4, 4),
        candidate_layers=layers,
        preprocessing_hash="p" * 64,
        split_hash="s" * 64,
        checkpoint_hash="c" * 64,
    )
    with pytest.raises(CacheManifestError, match="checkpoint_hash"):
        validate_sample_record(
            record,
            candidate_layers=layers,
            expected_checkpoint_hash="d" * 64,
            expected_split_hash="s" * 64,
            expected_preprocessing_hash="p" * 64,
        )


def test_validate_rejects_schema_version_mismatch():
    layers = (6, 12)
    record = build_sample_record(
        sample_id="a.png",
        label=0,
        mask_path="",
        category="bottle",
        split="train",
        maps=_toy_maps(layers),
        ingredients=_toy_ingredients(layers),
        teacher_logits=torch.randn(4, 4),
        candidate_layers=layers,
        preprocessing_hash="p" * 64,
        split_hash="s" * 64,
        checkpoint_hash="c" * 64,
    )
    record["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(CacheManifestError, match="schema_version"):
        validate_sample_record(record, candidate_layers=layers)


def test_write_shard_and_verify_roundtrip(tmp_path: Path):
    layers = (6, 12)
    records = [
        build_sample_record(
            sample_id=f"{i}.png",
            label=i % 2,
            mask_path="",
            category="bottle",
            split="train",
            maps=_toy_maps(layers),
            ingredients=_toy_ingredients(layers),
            teacher_logits=torch.randn(4, 4),
            candidate_layers=layers,
            preprocessing_hash="p" * 64,
            split_hash="s" * 64,
            checkpoint_hash="c" * 64,
        )
        for i in range(3)
    ]
    shard_path = tmp_path / "shard_0000.pt"
    write_shard(shard_path, records)
    loaded = verify_shard(
        shard_path,
        candidate_layers=layers,
        expected_checkpoint_hash="c" * 64,
        expected_split_hash="s" * 64,
        expected_preprocessing_hash="p" * 64,
    )
    assert len(loaded) == 3
    assert loaded[0]["sample_id"] == "0.png"
    assert loaded[0]["maps"][12][6].shape == (4, 4)


def test_verify_shard_rejects_stale_checkpoint_hash(tmp_path: Path):
    layers = (6, 12)
    record = build_sample_record(
        sample_id="a.png",
        label=0,
        mask_path="",
        category="bottle",
        split="train",
        maps=_toy_maps(layers),
        ingredients=_toy_ingredients(layers),
        teacher_logits=torch.randn(4, 4),
        candidate_layers=layers,
        preprocessing_hash="p" * 64,
        split_hash="s" * 64,
        checkpoint_hash="c" * 64,
    )
    shard_path = tmp_path / "shard_0000.pt"
    write_shard(shard_path, [record])
    with pytest.raises(CacheManifestError, match="checkpoint_hash"):
        verify_shard(
            shard_path,
            candidate_layers=layers,
            expected_checkpoint_hash="e" * 64,
            expected_split_hash="s" * 64,
            expected_preprocessing_hash="p" * 64,
        )


def test_parquet_index_lists_sample_locations(tmp_path: Path):
    rows = [
        {
            "sample_id": "a.png",
            "shard_name": "shard_0000.pt",
            "index_in_shard": 0,
            "label": 0,
            "split": "train",
            "category": "bottle",
        },
        {
            "sample_id": "b.png",
            "shard_name": "shard_0000.pt",
            "index_in_shard": 1,
            "label": 1,
            "split": "train",
            "category": "bottle",
        },
    ]
    index_path = tmp_path / "index.parquet"
    write_parquet_index(index_path, rows)
    assert index_path.is_file()
    import pyarrow.parquet as pq

    table = pq.read_table(index_path)
    assert table.num_rows == 2
    assert table.column("sample_id").to_pylist() == ["a.png", "b.png"]


def test_preprocessing_and_file_hash_are_stable(tmp_path: Path):
    path = tmp_path / "ckpt.bin"
    path.write_bytes(b"teacher-bytes")
    h1 = compute_file_hash(path)
    h2 = compute_file_hash(path)
    assert h1 == h2
    assert len(h1) == 64
    p1 = compute_preprocessing_hash(image_size=518, mean=(0.1, 0.2, 0.3), std=(0.4, 0.5, 0.6))
    p2 = compute_preprocessing_hash(image_size=518, mean=(0.1, 0.2, 0.3), std=(0.4, 0.5, 0.6))
    assert p1 == p2
    p3 = compute_preprocessing_hash(image_size=224, mean=(0.1, 0.2, 0.3), std=(0.4, 0.5, 0.6))
    assert p1 != p3


def test_cache_dataset_reads_by_sample_id(tmp_path: Path):
    from rad.data.cache_dataset import TeacherCacheDataset

    layers = (6, 12)
    records = [
        build_sample_record(
            sample_id=f"{i}.png",
            label=i % 2,
            mask_path="",
            category="bottle",
            split="train",
            maps=_toy_maps(layers),
            ingredients=_toy_ingredients(layers),
            teacher_logits=torch.randn(4, 4),
            candidate_layers=layers,
            preprocessing_hash="p" * 64,
            split_hash="s" * 64,
            checkpoint_hash="c" * 64,
        )
        for i in range(2)
    ]
    shard_path = tmp_path / "shard_0000.pt"
    write_shard(shard_path, records)
    write_parquet_index(
        tmp_path / "index.parquet",
        [
            {
                "sample_id": r["sample_id"],
                "shard_name": "shard_0000.pt",
                "index_in_shard": i,
                "label": r["label"],
                "split": r["split"],
                "category": r["category"],
            }
            for i, r in enumerate(records)
        ],
    )
    meta = {
        "schema_version": SCHEMA_VERSION,
        "candidate_layers": list(layers),
        "preprocessing_hash": "p" * 64,
        "split_hash": "s" * 64,
        "checkpoint_hash": "c" * 64,
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    ds = TeacherCacheDataset(tmp_path)
    assert len(ds) == 2
    item = ds[0]
    assert item["sample_id"] == "0.png"
    assert item["maps"][12][6].shape == (4, 4)

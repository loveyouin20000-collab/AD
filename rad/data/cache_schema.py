from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = 1


class CacheManifestError(ValueError):
    """Raised when a cache record or shard fails validation."""


def expected_map_layers(candidate_layers: Sequence[int]) -> dict[int, tuple[int, ...]]:
    layers = tuple(candidate_layers)
    return {
        depth: tuple(layer for layer in layers if layer <= depth) for depth in layers
    }


def compute_file_hash(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_preprocessing_hash(
    *,
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> str:
    payload = {
        "image_size": int(image_size),
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_sample_record(
    *,
    sample_id: str,
    label: int,
    mask_path: str,
    category: str,
    split: str,
    maps: Mapping[int, Mapping[int, torch.Tensor]],
    ingredients: Mapping[str, Any],
    teacher_logits: torch.Tensor,
    candidate_layers: Sequence[int],
    preprocessing_hash: str,
    split_hash: str,
    checkpoint_hash: str,
    schema_version: int = SCHEMA_VERSION,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "label": int(label),
        "mask_path": mask_path,
        "category": category,
        "split": split,
        "maps": {
            int(depth): {int(layer): tensor.detach().cpu() for layer, tensor in layer_maps.items()}
            for depth, layer_maps in maps.items()
        },
        "ingredients": _cpu_ingredients(ingredients),
        "teacher_logits": teacher_logits.detach().cpu(),
        "candidate_layers": [int(x) for x in candidate_layers],
        "preprocessing_hash": preprocessing_hash,
        "split_hash": split_hash,
        "checkpoint_hash": checkpoint_hash,
        "schema_version": int(schema_version),
    }


def _cpu_ingredients(ingredients: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in ingredients.items():
        if isinstance(value, Mapping):
            out[key] = {int(k): v.detach().cpu() for k, v in value.items()}
        else:
            out[key] = value
    return out


def validate_sample_record(
    record: Mapping[str, Any],
    *,
    candidate_layers: Sequence[int],
    expected_checkpoint_hash: str | None = None,
    expected_split_hash: str | None = None,
    expected_preprocessing_hash: str | None = None,
) -> None:
    if int(record.get("schema_version", -1)) != SCHEMA_VERSION:
        raise CacheManifestError(
            f"schema_version mismatch: got {record.get('schema_version')}, "
            f"expected {SCHEMA_VERSION}"
        )

    layers = tuple(int(x) for x in candidate_layers)
    if tuple(int(x) for x in record.get("candidate_layers", ())) != layers:
        raise CacheManifestError(
            f"candidate_layers mismatch: got {record.get('candidate_layers')}, expected {layers}"
        )

    for name, expected in (
        ("checkpoint_hash", expected_checkpoint_hash),
        ("split_hash", expected_split_hash),
        ("preprocessing_hash", expected_preprocessing_hash),
    ):
        if expected is not None and record.get(name) != expected:
            raise CacheManifestError(f"{name} mismatch: got {record.get(name)}, expected {expected}")

    maps = record.get("maps")
    if not isinstance(maps, Mapping):
        raise CacheManifestError("maps missing or invalid")

    expected = expected_map_layers(layers)
    for depth, layer_ids in expected.items():
        if depth not in maps:
            raise CacheManifestError(f"incomplete maps: missing checkpoint depth {depth}")
        layer_maps = maps[depth]
        for layer in layer_ids:
            if layer not in layer_maps:
                raise CacheManifestError(
                    f"incomplete maps: missing A_{{{layer}|{depth}}} at depth {depth}"
                )

    for key in ("teacher_logits", "ingredients", "sample_id", "label", "split"):
        if key not in record:
            raise CacheManifestError(f"missing field: {key}")


def write_shard(path: Path | str, records: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "records": list(records),
    }
    torch.save(payload, path)


def load_shard(path: Path | str) -> list[dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict) or "records" not in payload:
        raise CacheManifestError(f"invalid shard payload: {path}")
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise CacheManifestError(
            f"schema_version mismatch in shard {path}: "
            f"got {payload.get('schema_version')}, expected {SCHEMA_VERSION}"
        )
    records = payload["records"]
    if not isinstance(records, list):
        raise CacheManifestError(f"shard records must be a list: {path}")
    return records


def verify_shard(
    path: Path | str,
    *,
    candidate_layers: Sequence[int],
    expected_checkpoint_hash: str | None = None,
    expected_split_hash: str | None = None,
    expected_preprocessing_hash: str | None = None,
) -> list[dict[str, Any]]:
    records = load_shard(path)
    for record in records:
        validate_sample_record(
            record,
            candidate_layers=candidate_layers,
            expected_checkpoint_hash=expected_checkpoint_hash,
            expected_split_hash=expected_split_hash,
            expected_preprocessing_hash=expected_preprocessing_hash,
        )
    return records


def write_parquet_index(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([dict(row) for row in rows])
    pq.write_table(table, path)


def write_cache_meta(path: Path | str, meta: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(meta), indent=2, sort_keys=True) + "\n")


def iter_shard_paths(cache_dir: Path | str) -> Iterable[Path]:
    root = Path(cache_dir)
    return sorted(root.glob("shard_*.pt"))

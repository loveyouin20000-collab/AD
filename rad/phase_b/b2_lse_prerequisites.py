"""B2-06C LSE prerequisite materialization helpers."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, NoReturn

import torch

from rad.data.cache_schema import SCHEMA_VERSION, write_cache_meta, write_parquet_index, write_shard


class B2LSEPrerequisiteError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2LSEPrerequisiteError(code, detail)


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _load_json(path: Path, *, code: str) -> dict[str, Any]:
    if not path.is_file():
        _fail(code, f"missing {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        _fail("B2_LSE_PREREQ_JSON_INVALID", f"{path} must contain JSON object")
    return payload


def _load_record(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        _fail("B2_LSE_PREREQ_SAMPLE_INVALID", f"{path} must contain a dict")
    record = payload.get("record", payload)
    if not isinstance(record, dict):
        _fail("B2_LSE_PREREQ_SAMPLE_INVALID", f"{path} record must be a dict")
    for key in ("sample_id", "split", "candidate_layers", "checkpoint_hash", "split_hash"):
        if key not in record:
            _fail("B2_LSE_PREREQ_SAMPLE_INVALID", f"{path} missing {key}")
    return record


def _write_lse_cache(
    *,
    output_dir: Path,
    records: list[dict[str, Any]],
    split_name: str,
    manifest: dict[str, Any],
    data_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_name = "shard_00000.pt"
    write_shard(output_dir / shard_name, records)
    index_rows = [
        {
            "sample_id": str(record["sample_id"]),
            "shard_name": shard_name,
            "index_in_shard": i,
            "label": int(record["label"]),
            "split": split_name,
            "category": str(record["category"]),
        }
        for i, record in enumerate(records)
    ]
    write_parquet_index(output_dir / "index.parquet", index_rows)
    first = records[0]
    meta = {
        "schema_version": SCHEMA_VERSION,
        "split": split_name,
        "source_split": "training" if split_name == "train" else split_name,
        "n_records": len(records),
        "candidate_layers": [int(x) for x in manifest["candidate_layers"]],
        "checkpoint_hash": str(first["checkpoint_hash"]),
        "split_hash": str(first["split_hash"]),
        "preprocessing_hash": str(first.get("preprocessing_hash", "")),
        "data_path": str(data_path),
        "image_size": 518,
        "source_cache_scientific_sha256": str(manifest.get("cache_scientific_sha256", "")),
    }
    write_cache_meta(output_dir / "meta.json", meta)


def convert_b2_cache_for_lse(
    *,
    source_cache: Path | str,
    train_cache: Path | str,
    calibration_cache: Path | str,
    data_path: Path | str,
) -> dict[str, Any]:
    source = Path(source_cache)
    manifest = _load_json(
        source / "manifest.json",
        code="B2_LSE_PREREQ_SOURCE_CACHE_MANIFEST_MISSING",
    )
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        _fail("B2_LSE_PREREQ_SOURCE_CACHE_INVALID", "manifest samples must be a list")

    by_split: dict[str, list[dict[str, Any]]] = {"training": [], "calibration": [], "evaluation": []}
    for sample in samples:
        if not isinstance(sample, dict) or "relative_path" not in sample:
            _fail("B2_LSE_PREREQ_SOURCE_CACHE_INVALID", "sample entry missing relative_path")
        record = _load_record(source / str(sample["relative_path"]))
        split = str(record["split"])
        if split not in by_split:
            _fail("B2_LSE_PREREQ_SOURCE_CACHE_INVALID", f"unexpected split {split}")
        by_split[split].append(record)

    if not by_split["training"] or not by_split["calibration"]:
        _fail("B2_LSE_PREREQ_SOURCE_CACHE_INVALID", "training and calibration records are required")

    train_path = Path(train_cache)
    cal_path = Path(calibration_cache)
    _write_lse_cache(
        output_dir=train_path,
        records=by_split["training"],
        split_name="train",
        manifest=manifest,
        data_path=Path(data_path),
    )
    _write_lse_cache(
        output_dir=cal_path,
        records=by_split["calibration"],
        split_name="calibration",
        manifest=manifest,
        data_path=Path(data_path),
    )
    receipt: dict[str, Any] = {
        "schema_version": "b2_06c_lse_cache_conversion_receipt_v1",
        "source_cache": str(source.resolve()),
        "source_cache_scientific_sha256": str(manifest.get("cache_scientific_sha256", "")),
        "train_cache": str(train_path.resolve()),
        "calibration_cache": str(cal_path.resolve()),
        "training_count": len(by_split["training"]),
        "calibration_count": len(by_split["calibration"]),
        "evaluation_ignored_count": len(by_split["evaluation"]),
        "split_counts": dict(Counter(str(record["split"]) for records in by_split.values() for record in records)),
        "training_started": False,
        "lse_checkpoint_generated": False,
    }
    receipt["receipt_identity"] = canonical_json_sha256(dict(receipt))
    receipt_path = source.parent / "b2_06c_lse_cache_conversion_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (receipt_path.with_suffix(receipt_path.suffix + ".sha256")).write_text(
        sha256_file(receipt_path) + "  " + receipt_path.name + "\n",
        encoding="utf-8",
    )
    return receipt

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from VisualAD_lib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD  # noqa: E402
from rad.config import ExperimentConfig  # noqa: E402
from rad.data.cache_schema import (  # noqa: E402
    SCHEMA_VERSION,
    CacheManifestError,
    build_sample_record,
    compute_file_hash,
    compute_preprocessing_hash,
    verify_shard,
    write_cache_meta,
    write_parquet_index,
    write_shard,
)
from rad.data.teacher_inference import (  # noqa: E402
    build_causal_maps_and_ingredients,
    load_teacher_bundle,
)
from utils.transforms import get_transform  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache versioned teacher outputs")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "calibration"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_output(args: argparse.Namespace, cfg: ExperimentConfig) -> Path:
    if args.output is not None:
        return Path(args.output)
    if args.output_dir is not None:
        return Path(args.output_dir)
    dataset = cfg.data.dataset if cfg.data is not None else "dataset"
    return REPO_ROOT / "artifacts" / "cache" / f"{dataset}_{args.split}"


def load_split_rows(manifest: Path, split: str, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("split") != split:
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> int:
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    if cfg.data is None or cfg.teacher is None:
        raise SystemExit("config must define data.* and teacher.* for caching")

    seed = args.seed if args.seed is not None else cfg.seed
    torch.manual_seed(seed)
    device = args.device or cfg.device
    shard_size = args.shard_size if args.shard_size is not None else cfg.cache.shard_size
    if shard_size < 1:
        raise SystemExit("shard_size must be >= 1")

    output_dir = resolve_output(args, cfg)
    candidate_layers = cfg.backbone.candidate_layers
    split_manifest = Path(cfg.data.split_manifest)
    if not split_manifest.is_absolute():
        split_manifest = REPO_ROOT / split_manifest
    checkpoint_path = Path(cfg.teacher.checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path

    preprocessing_hash = compute_preprocessing_hash(
        image_size=cfg.image_size,
        mean=OPENAI_DATASET_MEAN,
        std=OPENAI_DATASET_STD,
    )
    split_hash = compute_file_hash(split_manifest)
    checkpoint_hash = compute_file_hash(checkpoint_path)
    rows = load_split_rows(split_manifest, args.split, args.limit)
    shards = chunked(rows, shard_size)

    print(f"config: {args.config}")
    print(f"seed: {seed}")
    print(f"split: {args.split}")
    print(f"samples: {len(rows)}")
    print(f"shards: {len(shards)} (size={shard_size})")
    print(f"output: {output_dir}")
    print(f"candidate_layers: {list(candidate_layers)}")
    print(f"preprocessing_hash: {preprocessing_hash[:12]}...")
    print(f"split_hash: {split_hash[:12]}...")
    print(f"checkpoint_hash: {checkpoint_hash[:12]}...")

    if args.dry_run:
        return 0

    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume and not args.force:
        raise SystemExit(
            f"output already exists: {output_dir} (pass --force to overwrite or --resume)"
        )
    if args.force and output_dir.exists():
        for path in output_dir.glob("shard_*.pt"):
            path.unlink()
        for name in ("index.parquet", "meta.json"):
            p = output_dir / name
            if p.exists():
                p.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "candidate_layers": list(candidate_layers),
        "preprocessing_hash": preprocessing_hash,
        "split_hash": split_hash,
        "checkpoint_hash": checkpoint_hash,
        "split": args.split,
        "seed": seed,
        "image_size": cfg.image_size,
        "teacher_checkpoint": str(checkpoint_path),
        "split_manifest": str(split_manifest),
        "dataset": cfg.data.dataset,
        "data_path": str(cfg.data.data_path),
    }

    if args.resume and (output_dir / "meta.json").is_file():
        existing = json.loads((output_dir / "meta.json").read_text())
        for key in ("preprocessing_hash", "split_hash", "checkpoint_hash", "schema_version"):
            if existing.get(key) != meta[key]:
                raise SystemExit(
                    f"resume blocked: {key} mismatch "
                    f"(existing={existing.get(key)} current={meta[key]})"
                )

    class _Args:
        image_size = cfg.image_size

    preprocess, _ = get_transform(_Args())
    bundle = load_teacher_bundle(
        checkpoint_path,
        device=device,
        backbone=cfg.teacher.backbone,
    )
    # Prefer config candidate layers (must be subset of teacher features).
    for layer in candidate_layers:
        if layer not in bundle.features_list:
            raise SystemExit(
                f"candidate layer {layer} not in teacher features_list {bundle.features_list}"
            )

    index_rows: list[dict[str, Any]] = []
    data_root = Path(cfg.data.data_path)

    for shard_idx, shard_rows in enumerate(tqdm(shards, desc="shards")):
        shard_name = f"shard_{shard_idx:04d}.pt"
        shard_path = output_dir / shard_name

        if args.resume and shard_path.is_file():
            try:
                verified = verify_shard(
                    shard_path,
                    candidate_layers=candidate_layers,
                    expected_checkpoint_hash=checkpoint_hash,
                    expected_split_hash=split_hash,
                    expected_preprocessing_hash=preprocessing_hash,
                )
                if len(verified) == len(shard_rows) and all(
                    verified[i]["sample_id"] == shard_rows[i]["sample_id"]
                    for i in range(len(shard_rows))
                ):
                    for i, record in enumerate(verified):
                        index_rows.append(
                            {
                                "sample_id": record["sample_id"],
                                "shard_name": shard_name,
                                "index_in_shard": i,
                                "label": record["label"],
                                "split": record["split"],
                                "category": record["category"],
                            }
                        )
                    continue
            except CacheManifestError:
                pass

        records: list[dict[str, Any]] = []
        for row in shard_rows:
            image_path = data_root / row["image_path"]
            image = Image.open(image_path).convert("RGB")
            tensor = preprocess(image).unsqueeze(0).to(bundle.device)
            maps, ingredients, teacher_logits = build_causal_maps_and_ingredients(
                bundle, tensor, candidate_layers
            )
            records.append(
                build_sample_record(
                    sample_id=row["sample_id"],
                    label=int(row["label"]),
                    mask_path=str(row.get("mask_path") or ""),
                    category=str(row["category"]),
                    split=str(row["split"]),
                    maps=maps,
                    ingredients=ingredients,
                    teacher_logits=teacher_logits,
                    candidate_layers=candidate_layers,
                    preprocessing_hash=preprocessing_hash,
                    split_hash=split_hash,
                    checkpoint_hash=checkpoint_hash,
                )
            )

        write_shard(shard_path, records)
        for i, record in enumerate(records):
            index_rows.append(
                {
                    "sample_id": record["sample_id"],
                    "shard_name": shard_name,
                    "index_in_shard": i,
                    "label": record["label"],
                    "split": record["split"],
                    "category": record["category"],
                }
            )

    write_parquet_index(output_dir / "index.parquet", index_rows)
    write_cache_meta(output_dir / "meta.json", meta)
    print(f"wrote {len(index_rows)} samples to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

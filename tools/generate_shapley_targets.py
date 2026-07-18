from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rad.config import ExperimentConfig  # noqa: E402
from rad.data.cache_dataset import TeacherCacheDataset  # noqa: E402
from rad.targets.shapley import (  # noqa: E402
    contributions_to_distribution,
    exact_shapley,
    expected_subset_count,
    localization_utility_factory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate exact Shapley contribution targets")
    parser.add_argument("--config", type=str, default="configs/rad/base.yaml")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tau-phi", type=float, default=1.0)
    parser.add_argument(
        "--depths",
        type=int,
        nargs="*",
        default=None,
        help="Checkpoint depths (default: all candidate layers except the first)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_mask(data_root: Path, mask_path: str, image_size: int) -> torch.Tensor:
    if not mask_path:
        return torch.zeros(image_size, image_size)
    path = data_root / mask_path
    if not path.is_file():
        return torch.zeros(image_size, image_size)
    img = Image.open(path).convert("L")
    t = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    mask = t(img)[0]
    return (mask > 0.5).float()


def main() -> int:
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    torch.manual_seed(seed)

    cache_dir = args.cache
    if cache_dir is None:
        if cfg.data is None:
            raise SystemExit("provide --cache or config data section")
        cache_dir = REPO_ROOT / "artifacts" / "cache" / "mvtec_teacher"
    cache_dir = Path(cache_dir)

    output = args.output
    if output is None:
        out_dir = args.output_dir or (REPO_ROOT / "artifacts" / "targets" / "shapley")
        output = Path(out_dir) / "shapley_targets.pt"
    output = Path(output)

    dataset = TeacherCacheDataset(cache_dir)
    layers = tuple(int(x) for x in dataset.meta["candidate_layers"])
    # Shapley at checkpoints that have >=2 layers available
    depths = args.depths if args.depths else [d for d in layers if d > layers[0]]
    image_size = int(dataset.meta.get("image_size", cfg.image_size))
    data_root = Path(dataset.meta.get("data_path", cfg.data.data_path if cfg.data else ""))

    print(f"cache: {cache_dir}")
    print(f"output: {output}")
    print(f"seed: {seed}")
    print(f"depths: {depths}")
    print(f"tau_phi: {args.tau_phi}")
    for d in depths:
        avail = tuple(x for x in layers if x <= d)
        print(f"  depth {d}: layers={avail} subsets={expected_subset_count(avail)}")

    n = len(dataset) if args.limit is None else min(len(dataset), args.limit)
    print(f"samples: {n}")
    if args.dry_run:
        return 0

    if output.exists() and not args.force:
        raise SystemExit(f"output exists: {output} (pass --force)")

    records: list[dict[str, Any]] = []
    for i in tqdm(range(n), desc="shapley"):
        sample = dataset[i]
        label = torch.tensor(float(sample["label"]))
        mask = load_mask(data_root, str(sample.get("mask_path") or ""), image_size)
        utility_fn = localization_utility_factory(mask, label)

        per_depth: dict[int, dict[str, torch.Tensor]] = {}
        for depth in depths:
            avail = [layer for layer in layers if layer <= depth]
            maps = torch.stack([sample["maps"][depth][layer] for layer in avail], dim=0)
            # Causal maps are already at image_size; ensure 2D
            if maps.ndim == 4:
                maps = maps[:, 0]
            phi = exact_shapley(maps, utility_fn)
            dist = contributions_to_distribution(phi, tau_phi=args.tau_phi)
            per_depth[int(depth)] = {"phi": phi.cpu(), "distribution": dist.cpu()}

        records.append(
            {
                "sample_id": sample["sample_id"],
                "label": int(sample["label"]),
                "category": sample["category"],
                "depths": per_depth,
            }
        )

    payload = {
        "schema_version": 1,
        "tau_phi": args.tau_phi,
        "candidate_layers": list(layers),
        "depths": list(depths),
        "seed": seed,
        "cache_meta": {
            "split_hash": dataset.meta.get("split_hash"),
            "checkpoint_hash": dataset.meta.get("checkpoint_hash"),
            "preprocessing_hash": dataset.meta.get("preprocessing_hash"),
        },
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    meta_path = output.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "n_records": len(records),
                "depths": list(depths),
                "tau_phi": args.tau_phi,
                "output": str(output),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote: {output} ({len(records)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

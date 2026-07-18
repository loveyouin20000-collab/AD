from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    data_path: Path


@dataclass(frozen=True)
class BaselineConfig:
    seed: int
    device: str
    backbone: str
    features_list: tuple[int, ...]
    epoch: int
    batch_size: int
    image_size: int
    train: DatasetSpec
    test: DatasetSpec
    output_dir: Path

    @classmethod
    def from_yaml(cls, path: Path) -> BaselineConfig:
        raw: dict[str, Any] = yaml.safe_load(path.read_text())
        return cls(
            seed=int(raw["seed"]),
            device=str(raw["device"]),
            backbone=str(raw["backbone"]),
            features_list=tuple(int(x) for x in raw["features_list"]),
            epoch=int(raw["epoch"]),
            batch_size=int(raw["batch_size"]),
            image_size=int(raw["image_size"]),
            train=DatasetSpec(
                dataset=str(raw["train"]["dataset"]),
                data_path=Path(raw["train"]["data_path"]),
            ),
            test=DatasetSpec(
                dataset=str(raw["test"]["dataset"]),
                data_path=Path(raw["test"]["data_path"]),
            ),
            output_dir=Path(raw["output_dir"]),
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def package_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
    }
    try:
        import torch

        versions["torch"] = torch.__version__
    except ImportError:
        versions["torch"] = "unavailable"
    return versions


def build_train_command(cfg: BaselineConfig, checkpoint_dir: Path) -> list[str]:
    features = [str(x) for x in cfg.features_list]
    return [
        sys.executable,
        str(REPO_ROOT / "train.py"),
        "--train_data_path",
        str(cfg.train.data_path),
        "--save_path",
        str(checkpoint_dir),
        "--train_dataset",
        cfg.train.dataset,
        "--backbone",
        cfg.backbone,
        "--features_list",
        *features,
        "--epoch",
        str(cfg.epoch),
        "--batch_size",
        str(cfg.batch_size),
        "--image_size",
        str(cfg.image_size),
        "--seed",
        str(cfg.seed),
        "--device",
        cfg.device,
    ]


def build_test_command(
    cfg: BaselineConfig,
    checkpoint_path: Path,
    result_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "test.py"),
        "--test_data_path",
        str(cfg.test.data_path),
        "--checkpoint_path",
        str(checkpoint_path),
        "--test_dataset",
        cfg.test.dataset,
        "--save_path",
        str(result_dir),
        "--seed",
        str(cfg.seed),
        "--device",
        cfg.device,
    ]


def format_command(cmd: list[str]) -> str:
    return " ".join(cmd)


def manifest_path(output_dir: Path) -> Path:
    return output_dir / "manifest.json"


def load_completed_manifest(output_dir: Path) -> dict[str, Any] | None:
    path = manifest_path(output_dir)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data.get("status") == "completed":
        return data
    return None


def validate_dataset_paths(cfg: BaselineConfig) -> None:
    missing = [
        p
        for p in (cfg.train.data_path, cfg.test.data_path)
        if not p.exists()
    ]
    if missing:
        joined = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"dataset path(s) not found: {joined}")


def write_manifest(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_path(output_dir)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_baseline(
    config_path: Path,
    *,
    seed: int | None = None,
    output_dir: Path | None = None,
    dry_run: bool = False,
    checkpoint: Path | None = None,
) -> int:
    cfg = BaselineConfig.from_yaml(config_path)
    if seed is not None:
        cfg = BaselineConfig(
            seed=seed,
            device=cfg.device,
            backbone=cfg.backbone,
            features_list=cfg.features_list,
            epoch=cfg.epoch,
            batch_size=cfg.batch_size,
            image_size=cfg.image_size,
            train=cfg.train,
            test=cfg.test,
            output_dir=cfg.output_dir,
        )
    if output_dir is not None:
        cfg = BaselineConfig(
            seed=cfg.seed,
            device=cfg.device,
            backbone=cfg.backbone,
            features_list=cfg.features_list,
            epoch=cfg.epoch,
            batch_size=cfg.batch_size,
            image_size=cfg.image_size,
            train=cfg.train,
            test=cfg.test,
            output_dir=output_dir,
        )

    checkpoint_dir = cfg.output_dir / "checkpoints"
    result_dir = cfg.output_dir / "results" / f"epoch_{cfg.epoch}"
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        train_cmd: list[str] | None = None
    else:
        checkpoint_path = checkpoint_dir / f"epoch_{cfg.epoch}.pth"
        train_cmd = build_train_command(cfg, checkpoint_dir)
    test_cmd = build_test_command(cfg, checkpoint_path, result_dir)

    print(f"config: {config_path}")
    print(f"output_dir: {cfg.output_dir}")
    print(f"train_data_path: {cfg.train.data_path}")
    print(f"test_data_path: {cfg.test.data_path}")
    if train_cmd is None:
        print("train_cmd: SKIPPED (using --checkpoint)")
    else:
        print(f"train_cmd: {format_command(train_cmd)}")
    print(f"test_cmd: {format_command(test_cmd)}")
    print(f"checkpoint_path: {checkpoint_path}")

    if dry_run:
        return 0

    validate_dataset_paths(cfg)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    existing = load_completed_manifest(cfg.output_dir)
    if existing is not None:
        raise FileExistsError(
            f"completed manifest already exists: {manifest_path(cfg.output_dir)}"
        )

    config_hashes = {str(config_path): sha256_file(config_path)}
    provenance: dict[str, Any] = {
        "status": "running",
        "commands": {
            "train": train_cmd,
            "test": test_cmd,
        },
        "git_sha": git_sha(),
        "versions": package_versions(),
        "config_hashes": config_hashes,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "metrics_path": str(result_dir),
        "seed": cfg.seed,
        "eval_only": train_cmd is None,
    }
    write_manifest(cfg.output_dir, provenance)

    if train_cmd is not None:
        subprocess.run(train_cmd, cwd=REPO_ROOT, check=True)
    subprocess.run(test_cmd, cwd=REPO_ROOT, check=True)

    provenance["status"] = "completed"
    write_manifest(cfg.output_dir, provenance)
    print(f"manifest written: {manifest_path(cfg.output_dir)}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce official VisualAD baseline")
    parser.add_argument("--config", required=True, help="path to baseline YAML config")
    parser.add_argument("--seed", type=int, default=None, help="override config seed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="override config output_dir",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="skip training and evaluate this checkpoint (official teacher freeze)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print resolved commands without running train/test",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_baseline(
        Path(args.config),
        seed=args.seed,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    raise SystemExit(main())

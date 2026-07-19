from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rad.artifacts import atomic_write_json, refuse_existing_run  # noqa: E402
from rad.errors import (  # noqa: E402
    ArtifactIntegrityError,
    DatasetIntegrityError,
    MetricComputationError,
    OutputProtectionError,
    RADContractError,
)

REQUIRED_METRIC_KEYS = (
    "image_auroc",
    "image_ap",
    "image_f1_max",
    "pixel_auroc",
    "pixel_ap",
    "pixel_f1_max",
    "pixel_aupro",
)

LOG_COLUMN_TO_METRIC_KEY = {
    "Pixel-AUROC": "pixel_auroc",
    "Pixel-F1": "pixel_f1_max",
    "Pixel-AP": "pixel_ap",
    "Pixel-AUPRO": "pixel_aupro",
    "Sample-AUROC": "image_auroc",
    "Sample-F1": "image_f1_max",
    "Sample-AP": "image_ap",
}


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


def validate_dataset_paths(cfg: BaselineConfig) -> None:
    missing = [
        p
        for p in (cfg.train.data_path, cfg.test.data_path)
        if not p.exists()
    ]
    if missing:
        joined = ", ".join(str(p) for p in missing)
        raise DatasetIntegrityError(f"dataset path(s) not found: {joined}")


def _split_pipe_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_visualad_log_metrics(log_path: Path) -> dict[str, float]:
    header_cells: list[str] | None = None
    mean_cells: list[str] | None = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if "Pixel-AUROC" in stripped and "Sample-AUROC" in stripped:
            header_cells = _split_pipe_row(stripped)
        elif stripped.startswith("| Mean") or "| Mean" in stripped:
            mean_cells = _split_pipe_row(stripped)
    if header_cells is None or mean_cells is None:
        raise MetricComputationError("Mean metrics row not found in log.txt")
    if len(header_cells) != len(mean_cells):
        raise MetricComputationError("log.txt header/mean column mismatch")

    metrics: dict[str, float] = {}
    for header, value_text in zip(header_cells[1:], mean_cells[1:], strict=True):
        metric_key = LOG_COLUMN_TO_METRIC_KEY.get(header)
        if metric_key is None:
            continue
        try:
            value = float(value_text)
        except ValueError as exc:
            raise MetricComputationError(
                f"invalid metric value for {header}: {value_text!r}"
            ) from exc
        metrics[metric_key] = value / 100.0
    return metrics


def validate_baseline_metrics(metrics: dict[str, Any]) -> None:
    for key in REQUIRED_METRIC_KEYS:
        if key not in metrics:
            raise MetricComputationError(f"missing required metric: {key}")
        value = metrics[key]
        if not isinstance(value, (int, float)):
            raise MetricComputationError(f"metric {key} is not numeric")
        if not math.isfinite(float(value)):
            raise MetricComputationError(f"nonfinite metric: {key}")


def load_or_materialize_metrics(result_dir: Path) -> dict[str, float]:
    metrics_path = result_dir / "metrics.json"
    if metrics_path.is_file():
        loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise MetricComputationError("metrics.json must contain an object")
        metrics = {key: float(loaded[key]) for key in REQUIRED_METRIC_KEYS if key in loaded}
        validate_baseline_metrics(metrics)
        return metrics

    log_path = result_dir / "log.txt"
    if not log_path.is_file():
        raise MetricComputationError("metrics.json and log.txt are both missing")
    metrics = parse_visualad_log_metrics(log_path)
    validate_baseline_metrics(metrics)
    atomic_write_json(metrics_path, metrics)
    return metrics


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
    try:
        refuse_existing_run(cfg.output_dir)
    except OutputProtectionError:
        raise

    eval_only = train_cmd is None
    if eval_only:
        if not checkpoint_path.is_file():
            raise ArtifactIntegrityError(f"checkpoint not found: {checkpoint_path}")
    elif train_cmd is not None:
        subprocess.run(train_cmd, cwd=REPO_ROOT, check=True)
        if not checkpoint_path.is_file():
            raise ArtifactIntegrityError(
                f"training completed but checkpoint not found: {checkpoint_path}"
            )

    checkpoint_sha256 = sha256_file(checkpoint_path)
    subprocess.run(test_cmd, cwd=REPO_ROOT, check=True)
    metrics = load_or_materialize_metrics(result_dir)

    config_hashes = {str(config_path.resolve()): sha256_file(config_path)}
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "commands": {
            "train": train_cmd,
            "test": test_cmd,
        },
        "git_sha": git_sha(),
        "versions": package_versions(),
        "config_path": str(config_path.resolve()),
        "config_hashes": config_hashes,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "metrics_path": str((result_dir / "metrics.json").resolve()),
        "seed": cfg.seed,
        "eval_only": eval_only,
        "dataset": cfg.test.dataset,
        "backbone": cfg.backbone,
        "candidate_layers": list(cfg.features_list),
        "metrics": metrics,
    }
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest_path(cfg.output_dir), provenance)
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
    try:
        return run_baseline(
            Path(args.config),
            seed=args.seed,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            checkpoint=args.checkpoint,
        )
    except RADContractError as exc:
        raise SystemExit(f"RAD contract error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())

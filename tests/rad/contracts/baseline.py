"""Baseline reproduction test contracts — assertions and fixtures only."""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path
from typing import Any

import yaml

REQUIRED_BASELINE_METRIC_KEYS = (
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


def write_minimal_baseline_config(
    path: Path,
    *,
    train_data: Path,
    test_data: Path,
    output_dir: Path,
    epoch: int = 2,
    seed: int = 111,
) -> Path:
    payload = {
        "seed": seed,
        "device": "cpu",
        "backbone": "ViT-L/14@336px",
        "features_list": [6, 12, 18, 24],
        "epoch": epoch,
        "batch_size": 2,
        "image_size": 518,
        "train": {"dataset": "mvtec", "data_path": str(train_data)},
        "test": {"dataset": "visa", "data_path": str(test_data)},
        "output_dir": str(output_dir),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def sample_visualad_log_txt(
    *,
    pixel_auroc: float = 95.8,
    pixel_f1: float = 34.6,
    pixel_ap: float = 28.4,
    pixel_aupro: float = 0.0,
    sample_auroc: float = 84.7,
    sample_f1: float = 82.5,
    sample_ap: float = 87.6,
) -> str:
    return textwrap.dedent(
        f"""\
        26-07-18 12:10:18.654 - INFO:
        | Class      |   Pixel-AUROC |   Pixel-F1 |   Pixel-AP |   Pixel-AUPRO |   Sample-AUROC |   Sample-F1 |   Sample-AP |
        |:-----------|--------------:|-----------:|-----------:|--------------:|---------------:|------------:|------------:|
        | candle     |          98.8 |       42.8 |       32.7 |             0 |           88.8 |        83.7 |        91.6 |
        | Mean       |          {pixel_auroc} |       {pixel_f1} |       {pixel_ap} |             {pixel_aupro} |           {sample_auroc} |        {sample_f1} |        {sample_ap} |
        """
    )


def write_sample_log_txt(result_dir: Path, **kwargs: float) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / "log.txt"
    log_path.write_text(sample_visualad_log_txt(**kwargs), encoding="utf-8")
    return log_path


def normalized_metrics_from_log_percentages(
    *,
    pixel_auroc: float = 95.8,
    pixel_f1: float = 34.6,
    pixel_ap: float = 28.4,
    pixel_aupro: float = 0.0,
    sample_auroc: float = 84.7,
    sample_f1: float = 82.5,
    sample_ap: float = 87.6,
) -> dict[str, float]:
    return {
        "pixel_auroc": pixel_auroc / 100.0,
        "pixel_f1_max": pixel_f1 / 100.0,
        "pixel_ap": pixel_ap / 100.0,
        "pixel_aupro": pixel_aupro / 100.0,
        "image_auroc": sample_auroc / 100.0,
        "image_f1_max": sample_f1 / 100.0,
        "image_ap": sample_ap / 100.0,
    }


def assert_dry_run_resolves_train_and_test(stdout: str) -> None:
    assert "train.py" in stdout
    assert "features_list 6 12 18 24" in stdout
    assert "test.py" in stdout


def assert_checkpoint_dry_run_skips_train(stdout: str, checkpoint: str) -> None:
    assert "SKIPPED (using --checkpoint)" in stdout
    assert checkpoint in stdout
    assert "test.py" in stdout


def assert_no_baseline_artifacts(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    assert list(output_dir.rglob("*")) == []


def assert_completed_baseline_manifest(manifest: dict[str, Any]) -> None:
    assert manifest.get("status") == "completed"
    for key in (
        "schema_version",
        "git_sha",
        "seed",
        "checkpoint_path",
        "checkpoint_sha256",
        "config_hashes",
        "commands",
        "versions",
    ):
        assert key in manifest, f"missing manifest key: {key}"
    assert manifest["schema_version"] == 1


def assert_required_metrics_finite(metrics: dict[str, Any]) -> None:
    for key in REQUIRED_BASELINE_METRIC_KEYS:
        assert key in metrics, f"missing metric key: {key}"
        value = metrics[key]
        assert isinstance(value, (int, float)), f"{key} is not numeric"
        assert math.isfinite(float(value)), f"{key} is not finite"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

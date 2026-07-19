from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rad.evaluation.paper_metrics import PaperMetrics
from rad.evaluation.zero_shot import compute_stratified_metrics, compute_transfer_metrics


@dataclass(frozen=True)
class TransferSamplePrediction:
    """Per-sample zero-shot transfer prediction record."""

    sample_id: str
    dataset: str
    selected_depth: int
    image_label: int
    residual_gain: float
    anomaly_area: float
    contrast_proxy: float
    boundary_complexity: float


def export_transfer_predictions(
    rows: Sequence[TransferSamplePrediction],
    *,
    output_dir: Path | str,
    full_depth: int,
    epsilon: float,
    adaptive_maps: np.ndarray | None = None,
    full_depth_maps: np.ndarray | None = None,
    masks: np.ndarray | None = None,
    images: np.ndarray | None = None,
    paper_metrics: PaperMetrics | None = None,
) -> dict[str, Any]:
    """Write per-sample predictions first, then aggregate summary.json."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "per_sample_predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row)) + "\n")

    depths = np.array([r.selected_depth for r in rows], dtype=np.int64)
    gains = np.array([r.residual_gain for r in rows], dtype=np.float64)
    labels = np.array([r.image_label for r in rows], dtype=np.int64)

    if adaptive_maps is not None and full_depth_maps is not None and masks is not None:
        summary = compute_transfer_metrics(
            adaptive_maps=adaptive_maps,
            full_depth_maps=full_depth_maps,
            masks=masks,
            selected_depths=depths,
            residual_gains=gains,
            image_labels=labels,
            epsilon=epsilon,
            full_depth=full_depth,
        )
        summary["depth_distribution"] = {
            str(k): int(v) for k, v in summary["exit_histogram"].items()
        }
        if images is not None:
            summary["stratified"] = compute_stratified_metrics(
                adaptive_maps=adaptive_maps,
                full_depth_maps=full_depth_maps,
                masks=masks,
                selected_depths=depths,
                residual_gains=gains,
                image_labels=labels,
                images=images,
                epsilon=epsilon,
                full_depth=full_depth,
            )
    else:
        summary = {
            "n": len(rows),
            "depth_distribution": {
                str(k): int(v)
                for k, v in zip(*np.unique(depths, return_counts=True), strict=True)
            }
            if len(depths)
            else {},
            "expected_depth": float(np.mean(depths)) if len(depths) else float("nan"),
            "false_safe_exit_rate": float(
                np.mean(gains[depths < full_depth] > epsilon)
                if np.any(depths < full_depth)
                else float("nan")
            ),
            "datasets": sorted({r.dataset for r in rows}),
        }
        if paper_metrics is not None:
            summary["paper_metrics"] = paper_metrics.as_dict()

    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rad.evaluation.zero_shot import compute_stratified_metrics, compute_transfer_metrics


@dataclass(frozen=True)
class TransferSamplePrediction:
    """Per-sample zero-shot transfer prediction record."""

    sample_id: str
    dataset: str
    selected_depth: int
    image_label: int
    residual_gain: float
    pixel_ap_adaptive: float
    pixel_ap_full: float
    pro_adaptive: float
    pro_full: float
    boundary_f_adaptive: float
    boundary_f_full: float
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

    if adaptive_maps is None:
        # Aggregate from per-sample scalar metrics only
        summary: dict[str, Any] = {
            "n": len(rows),
            "depth_distribution": {
                str(k): int(v)
                for k, v in zip(*np.unique(depths, return_counts=True))
            },
            "expected_depth": float(np.mean(depths)) if len(depths) else float("nan"),
            "false_safe_exit_rate": float(
                np.mean(gains[depths < full_depth] > epsilon)
                if np.any(depths < full_depth)
                else float("nan")
            ),
            "pixel_ap_drop": float(
                np.mean([r.pixel_ap_full - r.pixel_ap_adaptive for r in rows])
            )
            if rows
            else float("nan"),
            "pro_drop": float(np.mean([r.pro_full - r.pro_adaptive for r in rows]))
            if rows
            else float("nan"),
            "boundary_f_score_drop": float(
                np.mean([r.boundary_f_full - r.boundary_f_adaptive for r in rows])
            )
            if rows
            else float("nan"),
            "datasets": sorted({r.dataset for r in rows}),
        }
    else:
        assert full_depth_maps is not None and masks is not None
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

    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary

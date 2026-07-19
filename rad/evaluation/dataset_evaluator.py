"""Shared real-dataset evaluation loop (no aggregate paper metrics)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from rad.data.adapters.preprocess import (
    PreprocessSpec,
    preprocess_image,
    preprocess_mask,
)
from rad.data.adapters.types import EvaluationRecord
from rad.errors import DatasetIntegrityError, MetricComputationError
from rad.inference.adaptive_engine import AdaptiveResult
from rad.losses.localization import sample_localization_error


class SupportsAdaptiveInfer(Protocol):
    def infer(
        self,
        image: torch.Tensor,
        *,
        force_full_depth: bool = False,
        measure_timing: bool = False,
    ) -> AdaptiveResult: ...


@dataclass(frozen=True)
class SamplePrediction:
    sample_id: str
    dataset: str
    category: str
    split: str
    image_label: int
    image_score: float
    selected_depth: int
    residual_gain: float | None


@dataclass
class EvaluationOutputs:
    records: tuple[EvaluationRecord, ...]
    image_labels: np.ndarray
    image_scores: np.ndarray
    masks: np.ndarray
    anomaly_maps: np.ndarray
    selected_depths: np.ndarray
    residual_gains: np.ndarray | None
    sample_predictions: tuple[SamplePrediction, ...]


def _require_finite(name: str, sample_id: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise MetricComputationError(
            f"nonfinite {name} for sample_id={sample_id}"
        )


def _as_numpy_map(map_tensor: torch.Tensor) -> np.ndarray:
    arr = map_tensor.detach().float().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise MetricComputationError(f"expected 2D anomaly map, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _allocate_arrays(
    *,
    n: int,
    height: int,
    width: int,
    use_memmap: bool,
    memmap_dir: Path | None,
    with_residual: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
]:
    if use_memmap:
        if memmap_dir is None:
            raise DatasetIntegrityError("memmap_dir is required when use_memmap=True")
        memmap_dir.mkdir(parents=True, exist_ok=True)
        masks = np.lib.format.open_memmap(
            memmap_dir / "masks.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n, height, width),
        )
        anomaly_maps = np.lib.format.open_memmap(
            memmap_dir / "anomaly_maps.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n, height, width),
        )
    else:
        masks = np.zeros((n, height, width), dtype=np.float32)
        anomaly_maps = np.zeros((n, height, width), dtype=np.float32)

    image_labels = np.zeros((n,), dtype=np.int64)
    image_scores = np.zeros((n,), dtype=np.float64)
    selected_depths = np.zeros((n,), dtype=np.int64)
    residual_gains: np.ndarray | None
    if with_residual:
        residual_gains = np.zeros((n,), dtype=np.float64)
    else:
        residual_gains = None
    return masks, anomaly_maps, image_labels, image_scores, selected_depths, residual_gains


def evaluate_dataset(
    *,
    adapter: Any,
    engine: SupportsAdaptiveInfer,
    preprocess: PreprocessSpec,
    device: torch.device,
    split: str = "test",
    limit: int | None = None,
    force_full_depth: bool = False,
    compute_full_depth_reference: bool = False,
    use_memmap: bool = False,
    memmap_dir: Path | str | None = None,
) -> EvaluationOutputs:
    """Run adapter → preprocess → engine over a real dataset split.

    Residual gain (optional) uses ``sample_localization_error`` for both
    adaptive and full-depth logits. Aggregate paper metrics are intentionally
    outside this function.
    """
    records = list(adapter.records(split=split))
    records.sort(key=lambda r: r.sample_id)
    seen: set[str] = set()
    for record in records:
        if record.sample_id in seen:
            raise DatasetIntegrityError(f"duplicate sample_id: {record.sample_id}")
        seen.add(record.sample_id)
        if not record.image_path.is_file():
            raise DatasetIntegrityError(f"missing image: {record.image_path}")
        if record.image_label == 1 and (
            record.mask_path is None or not record.mask_path.is_file()
        ):
            raise DatasetIntegrityError(
                f"anomalous sample missing mask: {record.sample_id}"
            )

    if limit is not None:
        if int(limit) < 0:
            raise DatasetIntegrityError(f"limit must be >= 0, got {limit}")
        records = records[: int(limit)]

    n = len(records)
    h = w = int(preprocess.image_size)
    mm_dir = Path(memmap_dir) if memmap_dir is not None else None
    (
        masks,
        anomaly_maps,
        image_labels,
        image_scores,
        selected_depths,
        residual_gains,
    ) = _allocate_arrays(
        n=n,
        height=h,
        width=w,
        use_memmap=use_memmap,
        memmap_dir=mm_dir,
        with_residual=compute_full_depth_reference,
    )

    predictions: list[SamplePrediction] = []
    kept_records: list[EvaluationRecord] = []

    for index, record in enumerate(records):
        image = adapter.open_image(record)
        mask_img = adapter.open_mask(record)
        image_tensor = preprocess_image(image, preprocess).unsqueeze(0).to(device)
        mask_tensor = preprocess_mask(mask_img, preprocess.image_size)

        adaptive = engine.infer(
            image_tensor,
            force_full_depth=force_full_depth,
            measure_timing=False,
        )
        _require_finite("anomaly_map", record.sample_id, adaptive.final_map)
        _require_finite("image_score", record.sample_id, adaptive.image_score)

        map_np = _as_numpy_map(adaptive.final_map)
        if map_np.shape != (h, w):
            # Engine may return native resolution; require exact preprocess size.
            raise MetricComputationError(
                f"anomaly map shape {map_np.shape} != {(h, w)} for {record.sample_id}"
            )

        score = float(adaptive.image_score.reshape(-1)[0].item())
        depth = int(adaptive.selected_depth)
        gain_value: float | None = None

        if compute_full_depth_reference:
            full = engine.infer(
                image_tensor,
                force_full_depth=True,
                measure_timing=False,
            )
            _require_finite("full_anomaly_map", record.sample_id, full.final_map)

            adaptive_logits = adaptive.final_map
            full_logits = full.final_map
            if adaptive_logits.ndim == 2:
                adaptive_logits = adaptive_logits.unsqueeze(0)
            if full_logits.ndim == 2:
                full_logits = full_logits.unsqueeze(0)
            if adaptive_logits.ndim == 3:
                adaptive_logits = adaptive_logits.unsqueeze(1)
            if full_logits.ndim == 3:
                full_logits = full_logits.unsqueeze(1)

            mask_b = mask_tensor.to(device=device, dtype=adaptive_logits.dtype)
            if mask_b.ndim == 2:
                mask_b = mask_b.unsqueeze(0).unsqueeze(0)
            elif mask_b.ndim == 3:
                mask_b = mask_b.unsqueeze(0)

            label_b = torch.tensor(
                [record.image_label],
                device=device,
                dtype=adaptive_logits.dtype,
            )
            adaptive_error = sample_localization_error(
                adaptive_logits,
                mask_b,
                label_b,
            )
            full_error = sample_localization_error(
                full_logits,
                mask_b,
                label_b,
            )
            residual = torch.clamp(adaptive_error - full_error, min=0.0)
            _require_finite("residual_gain", record.sample_id, residual)
            gain_value = float(residual.reshape(-1)[0].item())
            assert residual_gains is not None
            residual_gains[index] = gain_value

        masks[index] = mask_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        anomaly_maps[index] = map_np
        image_labels[index] = int(record.image_label)
        image_scores[index] = score
        selected_depths[index] = depth
        kept_records.append(record)
        predictions.append(
            SamplePrediction(
                sample_id=record.sample_id,
                dataset=record.dataset,
                category=record.category,
                split=record.split,
                image_label=int(record.image_label),
                image_score=score,
                selected_depth=depth,
                residual_gain=gain_value,
            )
        )

    if use_memmap:
        # Ensure memmap contents are flushed.
        if hasattr(masks, "flush"):
            masks.flush()
        if hasattr(anomaly_maps, "flush"):
            anomaly_maps.flush()

    return EvaluationOutputs(
        records=tuple(kept_records),
        image_labels=np.asarray(image_labels),
        image_scores=np.asarray(image_scores),
        masks=np.asarray(masks),
        anomaly_maps=np.asarray(anomaly_maps),
        selected_depths=np.asarray(selected_depths),
        residual_gains=None if residual_gains is None else np.asarray(residual_gains),
        sample_predictions=tuple(predictions),
    )


__all__ = [
    "EvaluationOutputs",
    "SamplePrediction",
    "SupportsAdaptiveInfer",
    "evaluate_dataset",
]

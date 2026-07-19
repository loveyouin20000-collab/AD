"""Anomaly dataset adapters for paper evaluation."""

from __future__ import annotations

from rad.data.adapters.mvtec import MVTecAdapter
from rad.data.adapters.preprocess import (
    CLIP_MEAN,
    CLIP_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PreprocessSpec,
    build_preprocess,
    preprocess_image,
    preprocess_mask,
)
from rad.data.adapters.protocol import AnomalyDatasetAdapter
from rad.data.adapters.registry import (
    get_adapter,
    normalize_dataset_name,
    planned_unsupported_dataset_names,
    supported_dataset_names,
)
from rad.data.adapters.types import EvaluationRecord
from rad.data.adapters.visa import VisAAdapter

__all__ = [
    "AnomalyDatasetAdapter",
    "CLIP_MEAN",
    "CLIP_STD",
    "EvaluationRecord",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "MVTecAdapter",
    "PreprocessSpec",
    "VisAAdapter",
    "build_preprocess",
    "get_adapter",
    "normalize_dataset_name",
    "planned_unsupported_dataset_names",
    "preprocess_image",
    "preprocess_mask",
    "supported_dataset_names",
]

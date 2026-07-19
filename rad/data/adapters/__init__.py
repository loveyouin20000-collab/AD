"""Anomaly dataset adapters for paper evaluation."""

from __future__ import annotations

from rad.data.adapters.mvtec import MVTecAdapter
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
    "EvaluationRecord",
    "MVTecAdapter",
    "VisAAdapter",
    "get_adapter",
    "normalize_dataset_name",
    "planned_unsupported_dataset_names",
    "supported_dataset_names",
]

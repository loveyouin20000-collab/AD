"""Shared evaluation record types for anomaly dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluationRecord:
    sample_id: str
    dataset: str
    category: str
    image_path: Path
    mask_path: Path | None
    image_label: int
    split: str

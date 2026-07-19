"""Adapter protocol for anomaly evaluation datasets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from PIL import Image

from rad.data.adapters.types import EvaluationRecord


class AnomalyDatasetAdapter(Protocol):
    def records(self, split: str = "test") -> Sequence[EvaluationRecord]:
        ...

    def open_image(self, record: EvaluationRecord) -> Image.Image:
        ...

    def open_mask(self, record: EvaluationRecord) -> Image.Image | None:
        ...

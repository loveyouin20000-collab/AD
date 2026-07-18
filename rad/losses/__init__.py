"""Localization and distillation losses for RAD-VisualAD."""

from rad.losses.distillation import (
    confidence_weighted_distillation,
    normalized_binary_entropy,
)
from rad.losses.localization import sample_localization_error

__all__ = [
    "sample_localization_error",
    "confidence_weighted_distillation",
    "normalized_binary_entropy",
]

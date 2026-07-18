"""Offline supervision targets (Shapley contributions, residual gains)."""

from rad.targets.residual_gain import (
    build_gain_target_record,
    residual_gains,
    sufficiency,
)
from rad.targets.shapley import (
    contributions_to_distribution,
    exact_shapley,
    expected_subset_count,
    localization_utility_factory,
)

__all__ = [
    "exact_shapley",
    "contributions_to_distribution",
    "expected_subset_count",
    "localization_utility_factory",
    "residual_gains",
    "sufficiency",
    "build_gain_target_record",
]

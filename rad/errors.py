"""RAD contract exception hierarchy."""

from __future__ import annotations


class RADContractError(RuntimeError):
    """Base class for fail-closed RAD research contracts."""


class ConfigurationContractError(RADContractError):
    """Invalid or unsupported configuration."""


class OutputProtectionError(RADContractError):
    """Refused overwrite of an existing run directory or artifact."""


class DatasetIntegrityError(RADContractError):
    """Dataset layout, sample ID, image, or mask integrity failure."""


class ArtifactIntegrityError(RADContractError):
    """Missing, mismatched, or corrupt artifact/hash failure."""


class UnsupportedDatasetError(RADContractError):
    """Registered but unimplemented dataset adapter."""


class MetricComputationError(RADContractError):
    """Nonfinite, empty, or otherwise invalid metric computation."""


class ScientificGateError(RADContractError):
    """Scientific gate failed after a completed execution attempt."""

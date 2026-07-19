"""Dataset adapter registry and name normalization."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from rad.data.adapters.protocol import AnomalyDatasetAdapter
from rad.errors import ConfigurationContractError

_SUPPORTED: frozenset[str] = frozenset({"mvtec", "visa"})
_PLANNED_UNSUPPORTED: frozenset[str] = frozenset(
    {"btad", "ksdd2", "dagm", "dtd-synthetic"}
)

_ALIAS_TO_CANONICAL: dict[str, str] = {
    "mvtec": "mvtec",
    "mvtec_ad": "mvtec",
    "mvtec-ad": "mvtec",
    "mvtecad": "mvtec",
    "visa": "visa",
    "btad": "btad",
    "ksdd2": "ksdd2",
    "dagm": "dagm",
    "dtd-synthetic": "dtd-synthetic",
    "dtd_synthetic": "dtd-synthetic",
    "dtdsynthetic": "dtd-synthetic",
}


def _slug(name: str) -> str:
    lowered = name.strip().lower()
    lowered = lowered.replace("_", "-")
    lowered = re.sub(r"[\s]+", "-", lowered)
    lowered = re.sub(r"-ad$", "", lowered)
    lowered = lowered.replace("mvtec-ad", "mvtec")
    return lowered


def normalize_dataset_name(name: str) -> str:
    """Normalize a dataset name to a canonical registry key."""
    raw = name.strip()
    if not raw:
        raise ConfigurationContractError("Dataset name must be non-empty")

    slug = _slug(raw)
    # Collapse separators for alias lookup variants.
    compact = slug.replace("-", "").replace("_", "")
    candidates = (slug, slug.replace("-", "_"), compact)

    for candidate in candidates:
        if candidate in _ALIAS_TO_CANONICAL:
            return _ALIAS_TO_CANONICAL[candidate]
        # Direct supported / planned keys.
        if candidate in _SUPPORTED or candidate in _PLANNED_UNSUPPORTED:
            return candidate

    # Handle spaced forms already collapsed by _slug (e.g. "mvtec-ad" -> after -ad strip).
    if slug in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[slug]
    if slug in _SUPPORTED or slug in _PLANNED_UNSUPPORTED:
        return slug

    return slug


def supported_dataset_names() -> frozenset[str]:
    return _SUPPORTED


def planned_unsupported_dataset_names() -> frozenset[str]:
    return _PLANNED_UNSUPPORTED


def _load_mvtec(root: Path) -> AnomalyDatasetAdapter:
    from rad.data.adapters.mvtec import MVTecAdapter

    return MVTecAdapter(root)


def _load_visa(root: Path) -> AnomalyDatasetAdapter:
    from rad.data.adapters.visa import VisAAdapter

    return VisAAdapter(root)


_ADAPTER_FACTORIES: dict[str, Callable[[Path], AnomalyDatasetAdapter]] = {
    "mvtec": _load_mvtec,
    "visa": _load_visa,
}


def get_adapter(name: str, root: Path | str) -> AnomalyDatasetAdapter:
    """Resolve a dataset adapter by name.

    Only MVTec and VisA are production-supported in P0.
    Planned industrial datasets raise ``NotImplementedError``.
    Unknown names raise ``ConfigurationContractError``.
    """
    key = normalize_dataset_name(name)
    if key in _PLANNED_UNSUPPORTED:
        raise NotImplementedError(
            f"Dataset '{key}' is registered for the paper matrix but is not "
            "implemented in P0; only MVTec and VisA adapters are available"
        )
    if key not in _SUPPORTED:
        raise ConfigurationContractError(
            f"Unknown dataset '{name}' (normalized='{key}'). "
            f"Supported: {sorted(_SUPPORTED)}; "
            f"planned unsupported: {sorted(_PLANNED_UNSUPPORTED)}"
        )

    factory = _ADAPTER_FACTORIES[key]
    return factory(Path(root))

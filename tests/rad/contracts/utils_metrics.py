"""Load utils.metrics without importing the heavy utils package __init__."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_REPO = Path(__file__).resolve().parents[3]
_METRICS_PATH = _REPO / "utils" / "metrics.py"


def load_metrics_module() -> ModuleType:
    """Import cal_pro_score/compute_metrics without utils/__init__ side effects."""
    name = "_visualad_utils_metrics_isolated"
    spec = importlib.util.spec_from_file_location(name, _METRICS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load metrics from {_METRICS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

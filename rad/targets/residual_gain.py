from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F


def residual_gains(
    errors: Mapping[int, torch.Tensor],
    *,
    stop_gradient: bool = True,
    full_depth: int = 24,
    mid_depth: int = 18,
    early_depth: int = 12,
) -> dict[int, torch.Tensor]:
    """Compute residual localization gains from per-depth errors.

    g_mid = relu(E_mid - E_full)
    g_early = relu(E_early - min(E_mid, E_full))
    """
    required = (early_depth, mid_depth, full_depth)
    missing = [d for d in required if d not in errors]
    if missing:
        raise KeyError(f"errors missing depths {missing}; have {sorted(errors)}")

    e_early = errors[early_depth]
    e_mid = errors[mid_depth]
    e_full = errors[full_depth]
    if stop_gradient:
        e_early = e_early.detach()
        e_mid = e_mid.detach()
        e_full = e_full.detach()

    g_mid = F.relu(e_mid - e_full)
    g_early = F.relu(e_early - torch.minimum(e_mid, e_full))
    return {early_depth: g_early, mid_depth: g_mid}


def sufficiency(
    gain: torch.Tensor,
    error_at_depth: torch.Tensor,
    *,
    epsilon_gain: float,
    epsilon_absolute: float,
) -> torch.Tensor:
    """sufficient = (gain <= epsilon_gain) AND (E_d <= epsilon_absolute)."""
    return (gain <= epsilon_gain) & (error_at_depth <= epsilon_absolute)


def build_gain_target_record(
    errors: Mapping[int, torch.Tensor],
    *,
    epsilon_gain: float,
    epsilon_absolute: float,
    early_depths: Sequence[int] = (12, 18),
    full_depth: int = 24,
    stop_gradient: bool = True,
) -> dict[str, Any]:
    """Build a serializable target record with raw errors/gains for recalibration."""
    # Infer mid/early from early_depths + full_depth when standard (12,18,24)
    early_depths = tuple(int(d) for d in early_depths)
    if len(early_depths) < 1:
        raise ValueError("early_depths must be non-empty")

    # Prefer explicit mid = max(early_depths) when full is 24 and early includes 12,18
    mid_depth = max(early_depths)
    early_depth = min(early_depths)
    gains = residual_gains(
        errors,
        stop_gradient=stop_gradient,
        full_depth=full_depth,
        mid_depth=mid_depth,
        early_depth=early_depth,
    )

    raw_errors = {
        int(d): (v.detach() if stop_gradient else v).cpu()
        for d, v in errors.items()
    }
    raw_gains = {int(d): gains[d].detach().cpu() for d in early_depths if d in gains}
    # If early_depths has only one of 12/18, still include whatever residual_gains returned
    for d, g in gains.items():
        raw_gains.setdefault(int(d), g.detach().cpu())

    sufficient: dict[int, torch.Tensor] = {}
    for d in early_depths:
        if d not in raw_gains:
            continue
        err_d = raw_errors[d]
        suf = sufficiency(
            raw_gains[d],
            err_d,
            epsilon_gain=epsilon_gain,
            epsilon_absolute=epsilon_absolute,
        )
        sufficient[int(d)] = suf.cpu()

    return {
        "errors": raw_errors,
        "gains": raw_gains,
        "sufficient": sufficient,
        "epsilon_gain": float(epsilon_gain),
        "epsilon_absolute": float(epsilon_absolute),
        "early_depths": list(early_depths),
        "full_depth": int(full_depth),
    }

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations

import torch
import torch.nn.functional as F


def expected_subset_count(layer_ids: Sequence[int]) -> int:
    """Number of subsets enumerated for exact Shapley (2^L)."""
    return 1 << len(tuple(layer_ids))


def exact_shapley(
    layer_maps: torch.Tensor,
    utility_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Exact Shapley values for L layer maps under a set utility.

    Args:
        layer_maps: [L, H, W] (or [L, 1, H, W] squeezed)
        utility_fn: maps fused [H, W] -> scalar tensor
    Returns:
        phi: [L] signed contributions; sum(phi) == U(all) - U(empty)
    """
    if layer_maps.ndim == 4 and layer_maps.shape[1] == 1:
        layer_maps = layer_maps[:, 0]
    if layer_maps.ndim != 3:
        raise ValueError("layer_maps must have shape [L, H, W]")

    n_layers = int(layer_maps.shape[0])
    if n_layers == 0:
        return torch.zeros(0, device=layer_maps.device, dtype=layer_maps.dtype)

    # Precompute utility for every subset mask (bitmask 0 .. 2^L - 1)
    n_subsets = 1 << n_layers
    utilities = torch.empty(n_subsets, device=layer_maps.device, dtype=layer_maps.dtype)
    zero = torch.zeros_like(layer_maps[0])
    for mask in range(n_subsets):
        if mask == 0:
            fused = zero
        else:
            selected = [layer_maps[i] for i in range(n_layers) if mask & (1 << i)]
            fused = torch.stack(selected, dim=0).sum(dim=0)
        utilities[mask] = utility_fn(fused)

    phi = torch.zeros(n_layers, device=layer_maps.device, dtype=layer_maps.dtype)
    factorial = [1]
    for i in range(1, n_layers + 1):
        factorial.append(factorial[-1] * i)

    for i in range(n_layers):
        total = layer_maps.new_tensor(0.0)
        others = [j for j in range(n_layers) if j != i]
        for r in range(n_layers):
            # subsets of `others` with size r
            weight = factorial[r] * factorial[n_layers - r - 1] / factorial[n_layers]
            for subset in combinations(others, r):
                s_mask = 0
                for j in subset:
                    s_mask |= 1 << j
                with_i = s_mask | (1 << i)
                total = total + weight * (utilities[with_i] - utilities[s_mask])
        phi[i] = total

    return phi


def contributions_to_distribution(
    phi: torch.Tensor,
    tau_phi: float = 1.0,
) -> torch.Tensor:
    """Map signed Shapley values to a simplex via softmax(phi / tau)."""
    if tau_phi <= 0:
        raise ValueError("tau_phi must be positive")
    return F.softmax(phi / tau_phi, dim=-1)


def localization_utility_factory(
    mask: torch.Tensor,
    image_label: torch.Tensor,
):
    """Build U(fused)= -E_loc(fused, mask, label) for Shapley targets."""
    from rad.losses.localization import sample_localization_error

    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)
    if image_label.ndim == 0:
        image_label = image_label.unsqueeze(0)

    def utility_fn(fused: torch.Tensor) -> torch.Tensor:
        logits = fused.unsqueeze(0).unsqueeze(0)
        err = sample_localization_error(logits, mask, image_label)
        return -err[0]

    return utility_fn

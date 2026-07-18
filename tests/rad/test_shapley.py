from __future__ import annotations

import torch

from rad.targets.shapley import (
    contributions_to_distribution,
    exact_shapley,
    expected_subset_count,
)


def test_expected_subset_counts_match_checkpoint_layer_cardinality():
    # At d=12: layers {6,12} -> 2^2=4; d=18 -> 8; d=24 -> 16
    assert expected_subset_count((6, 12)) == 4
    assert expected_subset_count((6, 12, 18)) == 8
    assert expected_subset_count((6, 12, 18, 24)) == 16


def test_shapley_efficiency_sums_to_u_all_minus_u_empty():
    # Three synthetic maps; utility = mean of fused map
    maps = torch.stack(
        [
            torch.ones(4, 4) * 1.0,
            torch.ones(4, 4) * 2.0,
            torch.ones(4, 4) * 3.0,
        ],
        dim=0,
    )

    def utility_fn(fused: torch.Tensor) -> torch.Tensor:
        return fused.mean()

    phi = exact_shapley(maps, utility_fn)
    assert phi.shape == (3,)

    u_all = utility_fn(maps.sum(dim=0))
    u_empty = utility_fn(torch.zeros_like(maps[0]))
    assert torch.allclose(phi.sum(), u_all - u_empty, atol=1e-5)


def test_perfect_layer_receives_largest_contribution():
    h, w = 8, 8
    noise = torch.randn(h, w) * 0.01
    perfect = torch.zeros(h, w)
    perfect[2:6, 2:6] = 5.0  # strong localized signal
    weak = torch.randn(h, w) * 0.01
    maps = torch.stack([weak, perfect, noise], dim=0)
    target = perfect.clone()

    def utility_fn(fused: torch.Tensor) -> torch.Tensor:
        # Higher utility when fused matches the perfect pattern (neg L1)
        return -(fused - target).abs().mean()

    phi = exact_shapley(maps, utility_fn)
    assert int(phi.argmax().item()) == 1


def test_contributions_to_distribution_is_softmax():
    phi = torch.tensor([1.0, 2.0, 0.5])
    dist = contributions_to_distribution(phi, tau_phi=1.0)
    assert dist.shape == phi.shape
    assert torch.allclose(dist.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(dist, torch.softmax(phi / 1.0, dim=0), atol=1e-6)
    # Lower temperature sharpens
    sharp = contributions_to_distribution(phi, tau_phi=0.5)
    assert sharp[1] > dist[1]

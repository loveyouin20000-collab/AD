import pytest
import torch

from rad.types import StageCache


def test_stage_cache_rejects_invalid_next_block():
    with pytest.raises(ValueError):
        StageCache(sequence=torch.zeros(10, 2, 8), next_block=0, patch_tokens={})


def test_stage_cache_detach_removes_graph():
    cache = StageCache(
        sequence=torch.randn(10, 2, 8, requires_grad=True),
        next_block=7,
        patch_tokens={6: torch.randn(2, 7, 8, requires_grad=True)},
    )
    detached = cache.detach()
    assert not detached.sequence.requires_grad
    assert not detached.patch_tokens[6].requires_grad

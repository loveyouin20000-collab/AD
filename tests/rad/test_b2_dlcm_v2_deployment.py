"""RED/GREEN tests for B2-05C1 V2 deployment export."""

from __future__ import annotations

import pytest
import torch

from rad.phase_b import b2_dlcm_v2 as model_mod
from rad.phase_b import b2_dlcm_v2_deployment as subject
from tests.rad.b2_dlcm_fixtures import fixture_normalization_artifact


def _toy_checkpoint() -> dict:
    model = model_mod.B2DLCMV2(seed=17)
    return subject.export_v2_deployment_checkpoint(
        model,
        normalization_stats=fixture_normalization_artifact(),
        contribution_target_collection_scientific_sha256="bb" * 32,
    )


def test_export_gt_only_no_aux() -> None:
    ckpt = _toy_checkpoint()
    assert ckpt["auxiliary_heads_present"] is False
    for name in ckpt["state_dict"]:
        assert "teacher_allocation_head" not in name
        assert "gt_signed_head" not in name
        assert "teacher_signed_head" not in name
    assert any("gt_deployment_head" in name for name in ckpt["state_dict"])
    assert "H_deploy" in ckpt
    assert len(ckpt["golden_cases"]) == 9


def test_wrapper_blocks_auxiliary() -> None:
    wrapper = subject.load_v2_deployment_wrapper(_toy_checkpoint())
    with pytest.raises(subject.B2DLCMV2DeploymentError, match="B2_DLCM_V2_CONTRACT_MISMATCH"):
        wrapper.forward_diagnostic()
    weights = wrapper.forward(torch.zeros(1, 2, 18), depth=12, player_ids=[6, 12])
    assert weights.shape == (1, 2)

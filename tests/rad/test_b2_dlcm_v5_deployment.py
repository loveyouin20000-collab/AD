"""V5 deployment wrapper tests."""

from __future__ import annotations

import torch

from rad.phase_b import b2_dlcm_v5 as v5
from rad.phase_b import b2_dlcm_v5_deployment as deployment


def test_wrapper_beta1_reproduces_dynamic() -> None:
    dyn = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float32)
    state = {"w": torch.tensor([1.0, 2.0], dtype=torch.float32)}
    before = deployment.state_dict_tensor_digests(state)
    packed = deployment.wrap_c3_deployment_with_beta(
        dynamic_weight_fn=lambda: dyn.clone(),
        beta_index=100,
        checkpoint_state=state,
        h_deploy_v4="abc",
    )
    wrapper = packed["wrapper"]
    out = wrapper.deployment_weights()
    assert torch.equal(out, dyn)
    wrapper.verify_checkpoint_immutable(state)
    assert before == deployment.state_dict_tensor_digests(state)
    assert packed["category_in_wrapper"] is False


def test_wrapper_rejects_category_kw() -> None:
    dyn = torch.tensor([0.5, 0.5], dtype=torch.float32)
    packed = deployment.wrap_c3_deployment_with_beta(
        dynamic_weight_fn=lambda: dyn.clone(),
        beta_index=0,
    )
    import pytest

    with pytest.raises(deployment.B2DLCMV5DeploymentError) as exc:
        packed["wrapper"].deployment_weights(category="bottle")
    assert exc.value.code == "B2_DLCM_V5_CONTRACT_MISMATCH"


def test_h_deploy_binds_beta() -> None:
    h = v5.build_h_deploy_v5(
        h_deploy_v4="v4hash",
        beta_star_index=42,
        calibration_contract_identity=v5.v5_contract_identity(),
        calibration_ab_identity={"A": "x", "B": "x"},
    )
    assert isinstance(h, str) and len(h) == 64

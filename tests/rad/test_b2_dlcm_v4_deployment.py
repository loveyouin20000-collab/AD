"""V3 deployment tests."""

from __future__ import annotations

import pytest

from rad.phase_b import b2_dlcm_v4 as v3
from rad.phase_b import b2_dlcm_v4_deployment as deployment


def test_extract_drops_aux_heads() -> None:
    model = v3.B2DLCMV4(seed=17)
    state = v3.extract_deployment_state_dict(model)
    for prefix in v3.AUXILIARY_HEAD_PREFIXES:
        assert not any(k.startswith(prefix) for k in state)
    assert any(k.startswith("gt_deployment_head") for k in state)


def test_inference_forbids_aux_api() -> None:
    trunk = v3.B2DLCMV4DeploymentTrunk(seed=None, initialize=False)
    model = v3.B2DLCMV4(seed=17)
    trunk.load_state_dict(v3.extract_deployment_state_dict(model), strict=True)
    wrap = deployment.ImmutableV4DLCMInference(trunk)
    with pytest.raises(deployment.B2DLCMV4DeploymentError):
        _ = wrap.teacher_allocation_head


def test_h_deploy_pins_v3_architecture() -> None:
    h = deployment.build_h_deploy(
        {
            "deployment_state_scientific_sha256": "a" * 64,
            "descriptor_normalization_scientific_sha256": "b" * 64,
            "golden_cases_sha256": "c" * 64,
            "upstream": {},
        }
    )
    assert isinstance(h, str) and len(h) == 64
    assert v3.ARCHITECTURE_CONTRACT_VERSION == "b2_dlcm_architecture_v4"

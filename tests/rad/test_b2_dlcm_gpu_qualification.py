"""Dedicated GPU qualification entrypoint for future B2-05B.

CPU CI skips these tests with an explicit reason. When CUDA is visible they
exercise the §40 golden CPU→GPU numerical contract without starting real
authoritative training.
"""

from __future__ import annotations

import pytest
import torch

from rad.phase_b import b2_dlcm as model_mod
from rad.phase_b import b2_dlcm_deployment as subject
from tests.rad.b2_dlcm_fixtures import ACCEPTED_UPSTREAM, fixture_normalization_artifact

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for B2 DLCM GPU qualification entrypoint",
)


def _checkpoint() -> dict:
    model = model_mod.B2DLCM(seed=17)
    return subject.export_deployment_checkpoint(
        training_model=model,
        normalization=fixture_normalization_artifact(),
        canonical_seed=17,
        source_original_best_identity="aa" * 32,
        source_reproduction_best_identity="aa" * 32,
        contribution_target_collection_scientific_sha256=ACCEPTED_UPSTREAM[
            "contribution_target_collection_scientific_sha256"
        ],
    )


def test_gpu_golden_qualification_nine_cases() -> None:
    subject.clear_qualification_cache()
    ckpt = _checkpoint()
    device = torch.device("cuda:0")
    attestation = subject.run_gpu_qualification(
        ckpt,
        device=device,
        gpu_atol=1e-6,
    )
    assert attestation["cases_run"] == 9
    assert attestation["passed"] is True
    assert "max_logit_abs_error" in attestation
    assert "max_weight_abs_error" in attestation
    assert float(attestation["max_logit_abs_error"]) <= 1e-6
    assert float(attestation["max_weight_abs_error"]) <= 1e-6
    wrapper = subject.load_qualified_deployment(
        ckpt,
        checkpoint_file_sha256="cc" * 32,
        environment_contract_sha256="dd" * 32,
        device=device,
        gpu_uuid="gpu-qual-test",
    )
    raw = torch.zeros(2, 2, 18)
    weights = wrapper.forward(raw, 12, (6, 12))
    assert weights.device.type == "cuda"
    assert weights.dtype == torch.float32
    assert weights.shape == (2, 2)
    assert weights.requires_grad is False

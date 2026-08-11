"""B2-05C3 V3 deployment export/load (GT deployment head only)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NoReturn

import torch

from rad.phase_b import b2_dlcm as v1
from rad.phase_b import b2_dlcm_deployment as v1_deploy
from rad.phase_b import b2_dlcm_v4 as v3
from rad.phase_b import b2_dlcm_v4_protocol as protocol


class B2DLCMV4DeploymentError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMV4DeploymentError(code, detail)


def build_h_deploy(checkpoint: Mapping[str, Any]) -> str:
    payload = {
        "schema_version": "b2_dlcm_v4_h_deploy_v1",
        "architecture_contract_version": v3.ARCHITECTURE_CONTRACT_VERSION,
        "model_class_id": v3.MODEL_CLASS_ID,
        "deployment_state_scientific_sha256": checkpoint.get("deployment_state_scientific_sha256"),
        "descriptor_normalization_scientific_sha256": checkpoint.get(
            "descriptor_normalization_scientific_sha256"
        ),
        "golden_cases_sha256": checkpoint.get("golden_cases_sha256"),
        "upstream": checkpoint.get("upstream", {}),
    }
    return protocol.canonical_json_sha256(payload)


def export_v4_deployment_checkpoint(
    model: v3.B2DLCMV4,
    *,
    normalization_stats: Mapping[str, Any],
    contribution_target_collection_scientific_sha256: str,
    upstream: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state = v3.extract_deployment_state_dict(model)
    for banned in v3.AUXILIARY_HEAD_PREFIXES:
        if any(name.startswith(banned) for name in state):
            _fail("B2_DLCM_V4_CONTRACT_MISMATCH", f"auxiliary head leaked: {banned}")
    embedded = v1_deploy.embed_normalization(normalization_stats)
    trunk = v3.B2DLCMV4DeploymentTrunk(seed=None, initialize=False)
    trunk.load_state_dict(state, strict=True)
    golden = v1_deploy.generate_golden_cases(
        trunk,
        candidate_layers=model.candidate_layers,
        prediction_depths=model.prediction_depths,
    )
    golden_digest_payload = [
        {
            "case_id": case["case_id"],
            "prediction_depth": case["prediction_depth"],
            "input_tensor_sha256": case["input_tensor_sha256"],
            "expected_logits_bits": case["expected_logits_bits"],
            "expected_weights_bits": case["expected_weights_bits"],
        }
        for case in golden
    ]
    state_payload = {name: v1.tensor_sha256(tensor) for name, tensor in sorted(state.items())}
    checkpoint: dict[str, Any] = {
        "schema_version": "b2_dlcm_v4_deployment_checkpoint_v1",
        "architecture_contract_version": v3.ARCHITECTURE_CONTRACT_VERSION,
        "model_class_id": v3.MODEL_CLASS_ID,
        "candidate_layers": list(model.candidate_layers),
        "prediction_depths": list(model.prediction_depths),
        "state_dict": state,
        "deployment_state_scientific_sha256": protocol.canonical_json_sha256(state_payload),
        "descriptor_normalization": embedded,
        "descriptor_normalization_scientific_sha256": embedded.get(
            "descriptor_normalization_scientific_sha256"
        ),
        "contribution_target_collection_scientific_sha256": (
            contribution_target_collection_scientific_sha256
        ),
        "golden_cases": golden,
        "golden_cases_sha256": protocol.canonical_json_sha256(golden_digest_payload),
        "upstream": dict(upstream or {}),
        "category_in_checkpoint": False,
    }
    checkpoint["H_deploy"] = build_h_deploy(checkpoint)
    return checkpoint


class ImmutableV4DLCMInference:
    """Inference wrapper exposing only deployment weights."""

    def __init__(self, trunk: v3.B2DLCMV4DeploymentTrunk) -> None:
        self._trunk = trunk

    def forward(
        self,
        descriptors: torch.Tensor,
        *,
        prediction_depth: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._trunk.forward(descriptors, prediction_depth=prediction_depth)

    def __getattr__(self, name: str) -> Any:
        if name in {
            "teacher_allocation_head",
            "gt_signed_head",
            "teacher_signed_head",
            "forward_training",
        }:
            _fail("B2_DLCM_V4_CONTRACT_MISMATCH", f"auxiliary API forbidden: {name}")
        raise AttributeError(name)

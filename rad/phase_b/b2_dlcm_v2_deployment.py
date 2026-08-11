"""B2-05C1 V2 deployment export/load wrappers (GT deployment head only)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NoReturn

import torch

from rad.phase_b import b2_dlcm as v1
from rad.phase_b import b2_dlcm_deployment as v1_deploy
from rad.phase_b import b2_dlcm_v2 as v2
from rad.phase_b import b2_dlcm_v2_protocol as protocol


class B2DLCMV2DeploymentError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMV2DeploymentError(code, detail)


def build_h_deploy(checkpoint: Mapping[str, Any]) -> str:
    payload = {
        "schema_version": "b2_dlcm_v2_h_deploy_v1",
        "architecture_contract_version": v2.ARCHITECTURE_CONTRACT_VERSION,
        "model_class_id": v2.MODEL_CLASS_ID,
        "deployment_state_scientific_sha256": checkpoint.get("deployment_state_scientific_sha256"),
        "descriptor_normalization_scientific_sha256": checkpoint.get(
            "descriptor_normalization_scientific_sha256"
        ),
        "golden_cases_sha256": checkpoint.get("golden_cases_sha256"),
        "upstream": checkpoint.get("upstream", {}),
    }
    return protocol.canonical_json_sha256(payload)


def export_v2_deployment_checkpoint(
    model: v2.B2DLCMV2,
    *,
    normalization_stats: Mapping[str, Any],
    contribution_target_collection_scientific_sha256: str,
    upstream: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state = v2.extract_deployment_state_dict(model)
    for banned in v2.AUXILIARY_HEAD_PREFIXES:
        if any(name.startswith(banned) for name in state):
            _fail("B2_DLCM_V2_CONTRACT_MISMATCH", f"auxiliary head leaked: {banned}")
    embedded = v1_deploy.embed_normalization(normalization_stats)
    trunk = v2.B2DLCMV2DeploymentTrunk(seed=None, initialize=False)
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
    state_payload = {
        name: v1.tensor_sha256(tensor) for name, tensor in sorted(state.items())
    }
    checkpoint: dict[str, Any] = {
        "schema_version": "b2_dlcm_v2_deployment_checkpoint_v1",
        "architecture_contract_version": v2.ARCHITECTURE_CONTRACT_VERSION,
        "model_class_id": v2.MODEL_CLASS_ID,
        "candidate_layers": list(model.candidate_layers),
        "prediction_depths": list(model.prediction_depths),
        "state_dict": state,
        "deployment_state_scientific_sha256": protocol.canonical_json_sha256(state_payload),
        "descriptor_normalization_scientific_sha256": embedded[
            "descriptor_normalization_scientific_sha256"
        ],
        "embedded_normalization": embedded,
        "contribution_target_collection_scientific_sha256": (
            contribution_target_collection_scientific_sha256
        ),
        "golden_cases": golden,
        "golden_cases_sha256": protocol.canonical_json_sha256(golden_digest_payload),
        "upstream": dict(upstream or {}),
        "auxiliary_heads_present": False,
    }
    checkpoint["H_deploy"] = build_h_deploy(checkpoint)
    return checkpoint


class ImmutableV2DLCMInference:
    """Production wrapper exposing only deployment weights."""

    def __init__(self, trunk: v2.B2DLCMV2DeploymentTrunk, *, h_deploy: str) -> None:
        self._trunk = trunk
        self._trunk.eval()
        for param in self._trunk.parameters():
            param.requires_grad_(False)
        self.h_deploy = h_deploy

    @torch.inference_mode()
    def forward(
        self,
        raw_cpu_f32: torch.Tensor,
        *,
        depth: int,
        player_ids: list[int] | tuple[int, ...],
    ) -> torch.Tensor:
        if raw_cpu_f32.device.type != "cpu":
            _fail("B2_DLCM_V2_CONTRACT_MISMATCH", "formal forward expects CPU float32 input")
        logits, weights = self._trunk.forward(
            raw_cpu_f32,
            prediction_depth=int(depth),
            player_layer_ids=tuple(int(x) for x in player_ids),
        )
        _ = logits
        return weights

    def forward_diagnostic(self, *args: Any, **kwargs: Any) -> None:
        _fail(
            "B2_DLCM_V2_CONTRACT_MISMATCH",
            "auxiliary outputs unavailable from deployment wrapper",
        )


def load_v2_deployment_wrapper(checkpoint: Mapping[str, Any]) -> ImmutableV2DLCMInference:
    if checkpoint.get("auxiliary_heads_present") is True:
        _fail("B2_DLCM_V2_CONTRACT_MISMATCH", "deployment checkpoint must not include auxiliaries")
    trunk = v2.B2DLCMV2DeploymentTrunk(
        seed=None,
        initialize=False,
        candidate_layers=checkpoint["candidate_layers"],
        prediction_depths=checkpoint["prediction_depths"],
    )
    trunk.load_state_dict(checkpoint["state_dict"], strict=True)
    return ImmutableV2DLCMInference(trunk, h_deploy=str(checkpoint["H_deploy"]))

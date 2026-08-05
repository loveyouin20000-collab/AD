"""B2-05C4 V5 deployment: frozen C3 trunk + scalar beta wrapper."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NoReturn

import torch

from rad.phase_b import b2_dlcm as v1
from rad.phase_b import b2_dlcm_v5 as v5
from rad.phase_b import b2_dlcm_v5_protocol as protocol


class B2DLCMV5DeploymentError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMV5DeploymentError(code, detail)


def state_dict_tensor_digests(state: Mapping[str, torch.Tensor]) -> dict[str, str]:
    return {name: v1.tensor_sha256(tensor) for name, tensor in sorted(state.items())}


def assert_checkpoint_tensors_unchanged(
    before: Mapping[str, str],
    after_state: Mapping[str, torch.Tensor],
) -> None:
    after = state_dict_tensor_digests(after_state)
    if before != after:
        _fail("B2_DLCM_V5_CONTRACT_MISMATCH", "checkpoint tensors mutated by V5 wrapper")


class ImmutableV5DLCMInference:
    """Inference wrapper: C3 dynamic weights mixed with frozen scalar beta."""

    def __init__(
        self,
        *,
        dynamic_weight_fn: Any,
        beta: float,
        checkpoint_state: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        beta_f = float(beta)
        if not (0.0 <= beta_f <= 1.0) or beta_f != beta_f:
            _fail("B2_DLCM_V5_BETA_GRID_INVALID", f"beta out of [0,1]: {beta}")
        self._beta = beta_f
        self._dynamic_weight_fn = dynamic_weight_fn
        self._checkpoint_digests: dict[str, str] | None = None
        if checkpoint_state is not None:
            self._checkpoint_digests = state_dict_tensor_digests(checkpoint_state)

    @property
    def beta(self) -> float:
        return self._beta

    def deployment_weights(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if "category" in kwargs or "categories" in kwargs:
            _fail("B2_DLCM_V5_CONTRACT_MISMATCH", "category must not enter V5 wrapper")
        dynamic = self._dynamic_weight_fn(*args, **kwargs)
        if not isinstance(dynamic, torch.Tensor):
            dynamic = torch.tensor(dynamic, dtype=torch.float32)
        return v5.mix_uniform_anchored_weights(dynamic, self._beta)

    def verify_checkpoint_immutable(self, checkpoint_state: Mapping[str, torch.Tensor]) -> None:
        if self._checkpoint_digests is None:
            _fail("B2_DLCM_V5_CONTRACT_MISMATCH", "no baseline digests recorded")
        assert_checkpoint_tensors_unchanged(self._checkpoint_digests, checkpoint_state)


def wrap_c3_deployment_with_beta(
    *,
    dynamic_weight_fn: Any,
    beta_index: int,
    checkpoint_state: Mapping[str, torch.Tensor] | None = None,
    h_deploy_v4: str = "",
    calibration_contract_identity: Mapping[str, Any] | None = None,
    calibration_ab_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    beta = v5.beta_from_index(beta_index)
    wrapper = ImmutableV5DLCMInference(
        dynamic_weight_fn=dynamic_weight_fn,
        beta=beta,
        checkpoint_state=checkpoint_state,
    )
    h_deploy = v5.build_h_deploy_v5(
        h_deploy_v4=h_deploy_v4,
        beta_star_index=beta_index,
        calibration_contract_identity=calibration_contract_identity or v5.v5_contract_identity(),
        calibration_ab_identity=calibration_ab_identity or {},
    )
    return {
        "wrapper": wrapper,
        "beta_index": int(beta_index),
        "beta": beta,
        "beta_decimal": v5.beta_decimal_string(beta_index),
        "H_deploy": h_deploy,
        "category_in_wrapper": False,
        "checkpoint_tensors_mutated": False,
        "schema_version": "b2_dlcm_v5_deployment_wrapper_v1",
        "scientific_identity": protocol.canonical_json_sha256(
            {
                "H_deploy": h_deploy,
                "beta_index": int(beta_index),
                "beta_decimal": v5.beta_decimal_string(beta_index),
            }
        ),
    }

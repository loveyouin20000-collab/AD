"""B2-05C4 uniform-anchored DLCM V5: identity pins and FP32 weight mix."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, NoReturn

import torch
import torch.nn.functional as F

from rad.phase_b import b2_dlcm as v1
from rad.phase_b import b2_dlcm_v4 as v4

DEFAULT_CANDIDATE_LAYERS = v1.DEFAULT_CANDIDATE_LAYERS
DEFAULT_PREDICTION_DEPTHS = v1.DEFAULT_PREDICTION_DEPTHS
DEFAULT_DESCRIPTOR_DIMENSION = v1.DEFAULT_DESCRIPTOR_DIMENSION
DEFAULT_LAYER_EMBEDDING_DIM = v1.DEFAULT_LAYER_EMBEDDING_DIM
DEFAULT_DEPTH_EMBEDDING_DIM = v1.DEFAULT_DEPTH_EMBEDDING_DIM
DEFAULT_HIDDEN_DIMENSION = v1.DEFAULT_HIDDEN_DIMENSION
DEFAULT_DROPOUT_P = v1.DEFAULT_DROPOUT_P

ARCHITECTURE_CONTRACT_VERSION = "b2_dlcm_architecture_v5"
MODEL_CLASS_ID = "rad.phase_b.b2_dlcm_v5.B2DLCMV5"
CALIBRATION_CONTRACT_VERSION = "b2_dlcm_uniform_anchored_calibration_v5"
SCHEMA_VERSION = "b2_dlcm_uniform_anchored_contract_v5"

BETA_GRID_SIZE = 101
BETA_INDEX_MIN = 0
BETA_INDEX_MAX = 100
LOO_DEPTH = 24
GT_MACRO_MARGIN = 1e-5
GT_PER_CATEGORY_SLACK = 1e-4
LOO_TIE_EPS = 1e-5
TRAINING_CATEGORIES = ("bottle", "carpet")
CALIBRATION_PER_CATEGORY = 4

V2_CONTRACT_TAG = v4.V2_CONTRACT_TAG
V2_CONTRACT_COMMIT = v4.V2_CONTRACT_COMMIT
V3_CONTRACT_TAG = v4.V3_CONTRACT_TAG
V3_UNQUALIFIED_TAG = v4.V3_UNQUALIFIED_TAG
V3_UNQUALIFIED_COMMIT = v4.V3_UNQUALIFIED_COMMIT
V4_CONTRACT_TAG = "b2-dlcm-uniform-relative-contract-v4"
V4_CONTRACT_COMMIT = "3b2237affbe58a3cdc30d49bbdee4d8145a6a192"
V4_UNQUALIFIED_TAG = "b2-dlcm-uniform-relative-unqualified-evidence-v1"
V4_UNQUALIFIED_COMMIT = "a1447bdabdd7f54eb7883b717dfadc3da906da5b"
V4_ACCEPTED_PLAN_SHA256 = (
    "4979c73a28e0aaffd21f2c6408bb37e90fdc64201bcc326f990543fbbee5650f"
)
V4_ENVIRONMENT_IDENTITY = (
    "67677c4e9bb83475f7adc03294437bdd104a693e0465e107d3860096a9f03056"
)
V4_CANONICAL_SEED = 17
ADOPTED_ROSTER_SCIENTIFIC = v4.ADOPTED_ROSTER_SCIENTIFIC


class B2DLCMV5Error(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMV5Error(code, detail)


def v1_immutable_identity() -> dict[str, Any]:
    return v4.v1_immutable_identity()


def v2_immutable_identity() -> dict[str, Any]:
    return v4.v2_immutable_identity()


def v3_immutable_identity() -> dict[str, Any]:
    return v4.v3_immutable_identity()


def v4_immutable_identity() -> dict[str, Any]:
    return {
        "tag": V4_CONTRACT_TAG,
        "contract_commit": V4_CONTRACT_COMMIT,
        "unqualified_evidence_tag": V4_UNQUALIFIED_TAG,
        "unqualified_evidence_commit": V4_UNQUALIFIED_COMMIT,
        "accepted_plan_sha256": V4_ACCEPTED_PLAN_SHA256,
        "environment_identity": V4_ENVIRONMENT_IDENTITY,
        "canonical_seed": V4_CANONICAL_SEED,
        "adopted_final_roster_scientific_sha256": ADOPTED_ROSTER_SCIENTIFIC,
        "development_verdict": "development_unqualified",
        "failed_reason": "gt_category_kl:carpet",
    }


def beta_from_index(index: int) -> float:
    idx = int(index)
    if idx < BETA_INDEX_MIN or idx > BETA_INDEX_MAX:
        _fail("B2_DLCM_V5_BETA_GRID_INVALID", f"beta index out of range: {idx}")
    return idx / 100.0


def beta_decimal_string(index: int) -> str:
    idx = int(index)
    if idx < BETA_INDEX_MIN or idx > BETA_INDEX_MAX:
        _fail("B2_DLCM_V5_BETA_GRID_INVALID", f"beta index out of range: {idx}")
    return f"{idx / 100.0:.2f}"


def iter_beta_grid() -> Iterator[dict[str, Any]]:
    for index in range(BETA_GRID_SIZE):
        yield {
            "beta_index": index,
            "beta": beta_from_index(index),
            "beta_decimal": beta_decimal_string(index),
        }


def validate_beta_grid(indices: Sequence[int] | None = None) -> None:
    if indices is None:
        indices = list(range(BETA_GRID_SIZE))
    if len(indices) != BETA_GRID_SIZE:
        _fail("B2_DLCM_V5_BETA_GRID_INVALID", f"expected {BETA_GRID_SIZE} indices")
    if list(indices) != list(range(BETA_GRID_SIZE)):
        _fail("B2_DLCM_V5_BETA_GRID_INVALID", "beta grid must be contiguous 0..100")
    for item in iter_beta_grid():
        expected = item["beta_index"] / 100.0
        if float(item["beta"]) != expected:
            _fail("B2_DLCM_V5_BETA_GRID_INVALID", "beta != index/100.0")
        if item["beta_decimal"] != f"{expected:.2f}":
            _fail("B2_DLCM_V5_BETA_GRID_INVALID", "beta decimal string mismatch")


def depth_matched_uniform(
    n_players: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    if int(n_players) < 1:
        _fail("B2_DLCM_V5_CONTRACT_MISMATCH", f"invalid n_players={n_players}")
    device_t = device or torch.device("cpu")
    ref = v1.reference_uniform_weights(int(n_players), dtype=dtype).to(device=device_t)
    logits = torch.zeros(int(n_players), device=device_t, dtype=dtype)
    weights = F.softmax(logits, dim=0)
    if not bool(torch.equal(weights, ref)):
        _fail("B2_DLCM_V5_CONTRACT_MISMATCH", "uniform softmax != reference_uniform_weights")
    return weights


def mix_uniform_anchored_weights(
    dynamic_weights: torch.Tensor,
    beta: float,
    *,
    uniform: torch.Tensor | None = None,
) -> torch.Tensor:
    """FP32 convex mix: w = (1-beta)*u + beta*dynamic. Category never accepted."""

    if dynamic_weights.ndim not in (1, 2):
        _fail("B2_DLCM_V5_CONTRACT_MISMATCH", "dynamic_weights must be 1D or 2D")
    beta_f = float(beta)
    if not (0.0 <= beta_f <= 1.0) or beta_f != beta_f:
        _fail("B2_DLCM_V5_BETA_GRID_INVALID", f"beta out of [0,1]: {beta}")
    dyn = dynamic_weights.to(dtype=torch.float32)
    if uniform is None:
        n_players = int(dyn.shape[-1])
        uni = depth_matched_uniform(n_players, dtype=torch.float32, device=dyn.device)
        if dyn.ndim == 2:
            uni = uni.unsqueeze(0).expand_as(dyn)
    else:
        uni = uniform.to(dtype=torch.float32, device=dyn.device)
        if uni.shape != dyn.shape:
            if uni.ndim == 1 and dyn.ndim == 2 and uni.shape[0] == dyn.shape[-1]:
                uni = uni.unsqueeze(0).expand_as(dyn)
            else:
                _fail("B2_DLCM_V5_CONTRACT_MISMATCH", "uniform/dynamic shape mismatch")
    mixed = (1.0 - beta_f) * uni + beta_f * dyn
    if mixed.ndim == 1:
        total = mixed.sum()
        if not bool(torch.isfinite(total)) or float(total) <= 0.0:
            _fail("B2_DLCM_V5_CONTRACT_MISMATCH", "mixed weights invalid sum")
        if not bool(torch.all(mixed >= 0)):
            _fail("B2_DLCM_V5_CONTRACT_MISMATCH", "mixed weights must be non-negative")
        # Exact convex combo of normalized vectors stays normalized; assert tolerance.
        if abs(float(total) - 1.0) > 1e-6:
            mixed = mixed / total
        return mixed
    totals = mixed.sum(dim=-1, keepdim=True)
    if not bool(torch.isfinite(totals).all()) or bool((totals <= 0).any()):
        _fail("B2_DLCM_V5_CONTRACT_MISMATCH", "mixed weights invalid row sums")
    if not bool(torch.all(mixed >= 0)):
        _fail("B2_DLCM_V5_CONTRACT_MISMATCH", "mixed weights must be non-negative")
    if bool((torch.abs(totals - 1.0) > 1e-6).any()):
        mixed = mixed / totals
    return mixed


def per_sample_allocation_kl_from_weights(p: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Production exact KL(p || w) with zero-target masking (no epsilon)."""

    if p.shape != weights.shape:
        _fail("B2_DLCM_V5_CONTRACT_MISMATCH", "p and weights shapes must match")
    if p.requires_grad:
        _fail("B2_DLCM_V5_CONTRACT_MISMATCH", "allocation targets must not require grad")
    w = weights.to(dtype=torch.float32)
    if not bool(torch.all(w >= 0)):
        _fail("B2_DLCM_V5_CONTRACT_MISMATCH", "weights must be non-negative")
    # Match softmax KL path: log_w via log of positive weights; fail if non-positive mass.
    if bool((w <= 0).any()):
        # Allow exact zeros only where they never meet positive p (checked below).
        pass
    positive = p > 0
    if bool((positive & (w <= 0)).any()):
        _fail("B2_DLCM_V5_CONTRACT_MISMATCH", "positive target mass on non-positive weight")
    log_w = torch.where(w > 0, torch.log(w), torch.zeros_like(w))
    log_p = torch.where(positive, torch.log(torch.where(positive, p, torch.ones_like(p))), torch.zeros_like(p))
    kl_terms = torch.where(positive, p * (log_p - log_w), torch.zeros_like(p))
    return kl_terms.sum(dim=-1)


def build_h_deploy_v5(
    *,
    h_deploy_v4: str,
    beta_star_index: int,
    calibration_contract_identity: Mapping[str, Any],
    calibration_ab_identity: Mapping[str, Any],
) -> str:
    from rad.phase_b import b2_dlcm_v5_protocol as protocol

    payload = {
        "schema_version": "b2_dlcm_v5_h_deploy_v1",
        "architecture_contract_version": ARCHITECTURE_CONTRACT_VERSION,
        "model_class_id": MODEL_CLASS_ID,
        "h_deploy_v4": h_deploy_v4,
        "beta_star_index": int(beta_star_index),
        "beta_star_decimal": beta_decimal_string(int(beta_star_index)),
        "calibration_contract_identity": dict(calibration_contract_identity),
        "calibration_ab_identity": dict(calibration_ab_identity),
    }
    return protocol.canonical_json_sha256(payload)


def v5_contract_identity() -> dict[str, Any]:
    return {
        "architecture_contract_version": ARCHITECTURE_CONTRACT_VERSION,
        "model_class_id": MODEL_CLASS_ID,
        "calibration_contract_version": CALIBRATION_CONTRACT_VERSION,
        "schema_version": SCHEMA_VERSION,
    }

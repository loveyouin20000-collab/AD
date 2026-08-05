"""B2-05C3 uniform-relative DLCM V4: architecture pins, relative Smooth-Max GT loss.

Model architecture is identical to V2/V3 four-head decoupled DLCM. Category never
enters descriptors, embeddings, trunk, heads, deployment, or inference.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

import torch
import torch.nn as nn
import torch.nn.functional as F

from rad.phase_b import b2_dlcm as v1
from rad.phase_b import b2_dlcm_v2 as v2

DEFAULT_CANDIDATE_LAYERS = v1.DEFAULT_CANDIDATE_LAYERS
DEFAULT_PREDICTION_DEPTHS = v1.DEFAULT_PREDICTION_DEPTHS
DEFAULT_DESCRIPTOR_DIMENSION = v1.DEFAULT_DESCRIPTOR_DIMENSION
DEFAULT_LAYER_EMBEDDING_DIM = v1.DEFAULT_LAYER_EMBEDDING_DIM
DEFAULT_DEPTH_EMBEDDING_DIM = v1.DEFAULT_DEPTH_EMBEDDING_DIM
DEFAULT_HIDDEN_DIMENSION = v1.DEFAULT_HIDDEN_DIMENSION
DEFAULT_DROPOUT_P = v1.DEFAULT_DROPOUT_P

ARCHITECTURE_CONTRACT_VERSION = "b2_dlcm_architecture_v4"
MODEL_CLASS_ID = "rad.phase_b.b2_dlcm_v4.B2DLCMV4"
SMOOTHMAX_TAU = 0.05
TRAINING_CATEGORIES = ("bottle", "carpet")
GT_DEPLOYMENT_AGGREGATION = "uniform_relative_smooth_max"

TEACHER_ALLOC_WEIGHT = 0.25
GT_SIGNED_WEIGHT = 0.25
TEACHER_SIGNED_WEIGHT = 0.0625

AUXILIARY_HEAD_PREFIXES = v2.AUXILIARY_HEAD_PREFIXES

V2_CONTRACT_TAG = "b2-dlcm-decoupled-contract-v2"
V2_CONTRACT_COMMIT = "e54f2b44eeb962b05cfb7cf74764e55905f1a8f6"
V3_CONTRACT_TAG = "b2-dlcm-category-robust-contract-v3"
V3_UNQUALIFIED_TAG = "b2-dlcm-category-robust-unqualified-evidence-v1"
V3_UNQUALIFIED_COMMIT = "99c26de94ba7fa5358a7670473876c4a4cf1829d"
ADOPTED_ROSTER_SCIENTIFIC = (
    "267b7b527f13f84f76f69576d01b1532005d0bb7eda792d558ce5dcce1278213"
)


class B2DLCMV4Error(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMV4Error(code, detail)


def v1_immutable_identity() -> dict[str, Any]:
    return v2.v1_immutable_identity()


def v2_immutable_identity() -> dict[str, Any]:
    return {
        "tag": V2_CONTRACT_TAG,
        "commit": V2_CONTRACT_COMMIT,
        "adopted_final_roster_scientific_sha256": ADOPTED_ROSTER_SCIENTIFIC,
    }


def v3_immutable_identity() -> dict[str, Any]:
    return {
        "tag": V3_CONTRACT_TAG,
        "unqualified_evidence_tag": V3_UNQUALIFIED_TAG,
        "unqualified_evidence_commit": V3_UNQUALIFIED_COMMIT,
        "adopted_final_roster_scientific_sha256": ADOPTED_ROSTER_SCIENTIFIC,
        "development_verdict": "development_unqualified",
        "failed_reason": "gt_category_kl:carpet",
    }


class B2DLCMV4(v2.B2DLCMV2):
    """V4 model: identical architecture to V2/V3; distinct contract identity pins."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.architecture_contract_version = ARCHITECTURE_CONTRACT_VERSION
        self.model_class_id = MODEL_CLASS_ID


class B2DLCMV4DeploymentTrunk(v2.B2DLCMV2DeploymentTrunk):
    """Deployment trunk identical to V2/V3; V4 identity pins only."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.architecture_contract_version = ARCHITECTURE_CONTRACT_VERSION
        self.model_class_id = MODEL_CLASS_ID


def extract_deployment_state_dict(model: B2DLCMV4 | v2.B2DLCMV2) -> dict[str, torch.Tensor]:
    return v2.extract_deployment_state_dict(model)


def frozen_uniform_logits(
    batch: int,
    n_players: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Zero logits → FP32 softmax uniform; bit-match reference_uniform_weights."""

    if int(batch) < 1 or int(n_players) < 1:
        _fail("B2_DLCM_UNIFORM_BASELINE_INVALID", f"invalid shape batch={batch} n={n_players}")
    device_t = device or torch.device("cpu")
    logits = torch.zeros(int(batch), int(n_players), device=device_t, dtype=dtype)
    weights = F.softmax(logits, dim=-1)
    ref = v1.reference_uniform_weights(int(n_players), dtype=dtype).to(device=device_t)
    if not bool(torch.equal(weights[0], ref)):
        _fail("B2_DLCM_UNIFORM_BASELINE_INVALID", "zero-logit softmax != reference_uniform_weights")
    return logits


def relative_smooth_max_normalized(
    regrets: Sequence[float | torch.Tensor],
    *,
    tau: float = SMOOTHMAX_TAU,
) -> torch.Tensor:
    """Numerically stable normalized Smooth-Max over category relative regrets."""

    if float(tau) <= 0.0 or float(tau) != float(tau):
        _fail("B2_DLCM_RELATIVE_SMOOTHMAX_INVALID", f"tau must be positive finite, got {tau}")
    if len(regrets) < 1:
        _fail("B2_DLCM_RELATIVE_SMOOTHMAX_INVALID", "regrets must be non-empty")
    stacked = torch.stack(
        [r if isinstance(r, torch.Tensor) else torch.tensor(float(r)) for r in regrets]
    ).to(dtype=torch.float64)
    if not bool(torch.isfinite(stacked).all()):
        _fail("B2_DLCM_RELATIVE_SMOOTHMAX_INVALID", "category regrets must be finite")
    tau_t = torch.tensor(float(tau), dtype=torch.float64, device=stacked.device)
    m = stacked.max()
    # Equal regrets → exact common value (log(1)=0).
    if bool(torch.all(stacked == stacked[0])):
        return stacked[0].to(dtype=torch.float32)
    n = torch.tensor(float(stacked.numel()), dtype=torch.float64, device=stacked.device)
    robust = m + tau_t * torch.log((torch.exp((stacked - m) / tau_t).sum()) / n)
    if not bool(torch.isfinite(robust)):
        _fail("B2_DLCM_RELATIVE_SMOOTHMAX_INVALID", "relative smooth-max produced nonfinite value")
    return robust.to(dtype=torch.float32)


# Back-compat alias used by some tests that still call the Smooth-Max helper name.
smooth_max_normalized = relative_smooth_max_normalized


def per_sample_allocation_kl(p: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Per-sample target-weighted KL; same semantics as V1 without batch mean."""

    if p.shape != logits.shape:
        _fail("B2_DLCM_V4_CONTRACT_MISMATCH", "p and logits shapes must match")
    if p.requires_grad:
        _fail("B2_DLCM_V4_CONTRACT_MISMATCH", "allocation targets must not require grad")
    log_w = F.log_softmax(logits, dim=-1)
    positive = p > 0
    log_p = torch.zeros_like(p)
    log_p = torch.where(positive, torch.log(torch.where(positive, p, torch.ones_like(p))), log_p)
    kl_terms = torch.where(positive, p * (log_p - log_w), torch.zeros_like(p))
    return kl_terms.sum(dim=-1)


def category_mean_allocation_kl(
    p: torch.Tensor,
    logits: torch.Tensor,
    categories: Sequence[str],
    *,
    expected: Sequence[str] = TRAINING_CATEGORIES,
    require_exact_counts: Mapping[str, int] | None = None,
) -> dict[str, torch.Tensor]:
    """Mean allocation KL within each expected category."""

    if len(categories) != int(p.shape[0]):
        _fail("B2_DLCM_CATEGORY_BATCH_INVALID", "categories length must match batch")
    per_sample = per_sample_allocation_kl(p, logits)
    out: dict[str, torch.Tensor] = {}
    for cat in expected:
        mask = [i for i, c in enumerate(categories) if str(c) == cat]
        if not mask:
            _fail("B2_DLCM_CATEGORY_COVERAGE_INVALID", f"missing category {cat}")
        if require_exact_counts is not None and len(mask) != int(require_exact_counts[cat]):
            _fail(
                "B2_DLCM_CATEGORY_BATCH_INVALID",
                f"category {cat} count {len(mask)} != {require_exact_counts[cat]}",
            )
        idx = torch.tensor(mask, dtype=torch.long, device=per_sample.device)
        out[cat] = per_sample.index_select(0, idx).mean()
    return out


def batch_matched_relative_regrets(
    p: torch.Tensor,
    model_logits: torch.Tensor,
    categories: Sequence[str],
    *,
    expected: Sequence[str] = TRAINING_CATEGORIES,
    require_exact_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Batch-matched model/uniform category KLs and relative regrets R=K_m-K_u."""

    if p.shape != model_logits.shape:
        _fail("B2_DLCM_RELATIVE_REGRET_INVALID", "p and model logits shapes must match")
    uni_logits = frozen_uniform_logits(
        int(p.shape[0]),
        int(p.shape[-1]),
        device=p.device,
        dtype=torch.float32 if p.dtype == torch.float32 else p.dtype,
    )
    # Uniform path must not receive gradients.
    uni_logits = uni_logits.detach()
    model_kl = category_mean_allocation_kl(
        p,
        model_logits,
        categories,
        expected=expected,
        require_exact_counts=require_exact_counts,
    )
    uniform_kl = category_mean_allocation_kl(
        p,
        uni_logits,
        categories,
        expected=expected,
        require_exact_counts=require_exact_counts,
    )
    regrets: dict[str, torch.Tensor] = {}
    for cat in expected:
        # Direct subtraction — no slack, no clamp, no abs; preserve IEEE bits/sign.
        regrets[cat] = model_kl[cat] - uniform_kl[cat]
        if not bool(torch.isfinite(regrets[cat])):
            _fail("B2_DLCM_RELATIVE_REGRET_INVALID", f"nonfinite regret for {cat}")
    return {
        "model_kl": model_kl,
        "uniform_kl": uniform_kl,
        "regrets": regrets,
        "uniform_logits": uni_logits,
    }


def total_dlcm_v4_loss(
    depth_batch: Mapping[int, Mapping[str, torch.Tensor]],
    *,
    categories: Sequence[str],
    tau: float = SMOOTHMAX_TAU,
    teacher_alloc_weight: float = TEACHER_ALLOC_WEIGHT,
    gt_signed_weight: float = GT_SIGNED_WEIGHT,
    teacher_signed_weight: float = TEACHER_SIGNED_WEIGHT,
    ranking_weight: float = 0.25,
    huber_delta: float = 1.0,
    tie_tolerance: float = 1e-6,
    depth_weights: Mapping[int, float] | None = None,
    require_batch_category_counts: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """V4 total loss: relative Smooth-Max GT deploy + sample-mean auxiliaries."""

    if depth_weights is None:
        depth_weights = {12: 1 / 3, 18: 1 / 3, 24: 1 / 3}
    depths = sorted(depth_batch)
    if not depths:
        _fail("B2_DLCM_V4_CONTRACT_MISMATCH", "no depths provided")
    require_counts = {"bottle": 2, "carpet": 2} if require_batch_category_counts else None
    total: torch.Tensor | None = None
    depth_parts: dict[int, Any] = {}
    for depth in depths:
        payload = depth_batch[depth]
        matched = batch_matched_relative_regrets(
            payload["p_gt"],
            payload["gt_deployment_logits"],
            categories,
            expected=TRAINING_CATEGORIES,
            require_exact_counts=require_counts,
        )
        regrets = matched["regrets"]
        gt_relative = _relative_smooth_max_from_tensors(
            regrets["bottle"],
            regrets["carpet"],
            tau=float(tau),
        )
        t_alloc = v1.allocation_kl(payload["p_t"], payload["teacher_allocation_logits"])
        s_gt, gt_parts = v1.signed_loss(
            payload["gt_signed"],
            payload["phi_gt"],
            huber_delta=huber_delta,
            ranking_weight=ranking_weight,
            tie_tolerance=tie_tolerance,
        )
        s_t, t_parts = v1.signed_loss(
            payload["teacher_signed"],
            payload["phi_t"],
            huber_delta=huber_delta,
            ranking_weight=ranking_weight,
            tie_tolerance=tie_tolerance,
        )
        depth_loss = (
            gt_relative
            + float(gt_signed_weight) * s_gt
            + float(teacher_alloc_weight) * t_alloc
            + float(teacher_signed_weight) * s_t
        )
        weight = float(depth_weights[depth])
        total = depth_loss * weight if total is None else total + depth_loss * weight
        depth_parts[depth] = {
            "loss": depth_loss,
            "gt_deploy_relative_smooth_max": gt_relative,
            "gt_deploy_kl_bottle": matched["model_kl"]["bottle"],
            "gt_deploy_kl_carpet": matched["model_kl"]["carpet"],
            "gt_uniform_kl_bottle": matched["uniform_kl"]["bottle"],
            "gt_uniform_kl_carpet": matched["uniform_kl"]["carpet"],
            "gt_relative_regret_bottle": regrets["bottle"],
            "gt_relative_regret_carpet": regrets["carpet"],
            "teacher_alloc_kl": t_alloc,
            "gt_signed": gt_parts,
            "teacher_signed": t_parts,
            "weight": weight,
        }
    assert total is not None
    return total, {
        "depths": depth_parts,
        "depth_weights": dict(depth_weights),
        "tau": float(tau),
        "aggregation": {
            "gt_deployment": GT_DEPLOYMENT_AGGREGATION,
            "teacher_allocation": "sample_mean",
            "gt_signed": "sample_mean",
            "teacher_signed": "sample_mean",
        },
    }


def _relative_smooth_max_from_tensors(
    regret_b: torch.Tensor,
    regret_c: torch.Tensor,
    *,
    tau: float,
) -> torch.Tensor:
    if float(tau) <= 0.0:
        _fail("B2_DLCM_RELATIVE_SMOOTHMAX_INVALID", f"tau must be positive, got {tau}")
    # Equal → exact common value while preserving dtype/device/grad.
    if bool((regret_b - regret_c).abs() <= torch.finfo(torch.float32).eps * 8):
        return 0.5 * (regret_b + regret_c)
    m = torch.maximum(regret_b, regret_c)
    tau_t = regret_b.new_tensor(float(tau))
    return m + tau_t * torch.log(
        (torch.exp((regret_b - m) / tau_t) + torch.exp((regret_c - m) / tau_t))
        / regret_b.new_tensor(2.0)
    )


def model_state_scientific_sha256(model: nn.Module) -> str:
    return v1.model_state_scientific_sha256(model)

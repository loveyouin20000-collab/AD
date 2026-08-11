"""B2-05C1 V2 evaluation gates, auxiliary diagnostics, and decision identities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NoReturn

import torch

from rad.phase_b import b2_dlcm_evaluation as v1_eval
from rad.phase_b import b2_dlcm_v2_protocol as protocol
from rad.phase_b.b2_dlcm_deployment import (
    allocation_jsd,
    spearman_average_ranks,
    top1_set_agreement,
)

GT_MACRO_MARGIN = 1e-5
GT_PER_CATEGORY_SLACK = 1e-4
LOC_MACRO_PIXEL_AP_MIN = 0.0
LOC_MACRO_PIXEL_AUROC_MIN = -1e-4
LOC_MACRO_AUPRO_MIN = -1e-4
LOC_PER_CATEGORY_FLOOR = -1e-3


class B2DLCMV2EvaluationError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMV2EvaluationError(code, detail)


def evaluate_development_gates(
    *,
    depth24_gt_kl_macro: float,
    depth24_uniform_gt_kl_macro: float,
    per_category_gt_kl: Mapping[str, float],
    per_category_uniform_gt_kl: Mapping[str, float],
    delta_pixel_ap_macro: float,
    delta_pixel_auroc_macro: float,
    delta_aupro_macro: float,
    per_category_localization: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Depth-24 blocking gates for development (identical thresholds for final)."""

    reasons: list[str] = []
    if not (depth24_gt_kl_macro <= depth24_uniform_gt_kl_macro - GT_MACRO_MARGIN):
        reasons.append("gt_macro_kl")
    for cat, kl in per_category_gt_kl.items():
        uni = float(per_category_uniform_gt_kl[cat])
        if not (float(kl) <= uni + GT_PER_CATEGORY_SLACK):
            reasons.append(f"gt_category_kl:{cat}")
    if not (delta_pixel_ap_macro >= LOC_MACRO_PIXEL_AP_MIN):
        reasons.append("delta_pixel_ap_macro")
    if not (delta_pixel_auroc_macro >= LOC_MACRO_PIXEL_AUROC_MIN):
        reasons.append("delta_pixel_auroc_macro")
    if not (delta_aupro_macro >= LOC_MACRO_AUPRO_MIN):
        reasons.append("delta_aupro_macro")
    for cat, metrics in per_category_localization.items():
        for key in ("delta_pixel_ap", "delta_pixel_auroc", "delta_aupro"):
            if float(metrics[key]) < LOC_PER_CATEGORY_FLOOR:
                reasons.append(f"loc_category:{cat}:{key}")
    passed = len(reasons) == 0
    return {
        "passed": passed,
        "qualification_blocking": True,
        "failed_reasons": reasons,
        "thresholds": {
            "gt_macro_margin": GT_MACRO_MARGIN,
            "gt_per_category_slack": GT_PER_CATEGORY_SLACK,
            "loc_macro": {
                "delta_pixel_ap": LOC_MACRO_PIXEL_AP_MIN,
                "delta_pixel_auroc": LOC_MACRO_PIXEL_AUROC_MIN,
                "delta_aupro": LOC_MACRO_AUPRO_MIN,
            },
            "loc_per_category_floor": LOC_PER_CATEGORY_FLOOR,
        },
    }


def require_development_go(result: Mapping[str, Any]) -> None:
    if result.get("passed") is not True:
        _fail(
            "B2_DLCM_DEVELOPMENT_UNQUALIFIED",
            f"development failed: {result.get('failed_reasons')}",
        )


def build_auxiliary_diagnostics_manifest(
    *,
    diagnostics: Mapping[str, Any],
    source_checkpoint_kind: str,
) -> dict[str, Any]:
    if source_checkpoint_kind != "canonical_best_training_checkpoint":
        _fail(
            "B2_DLCM_AUXILIARY_DIAGNOSTICS_INVALID",
            "diagnostics must come from canonical_best_training_checkpoint",
        )
    if not diagnostics:
        _fail("B2_DLCM_AUXILIARY_DIAGNOSTICS_INVALID", "diagnostics payload empty")

    def _finite(value: Any) -> bool:
        if isinstance(value, bool):
            return True
        if isinstance(value, int | float):
            return value == value and value not in (float("inf"), float("-inf"))
        if isinstance(value, Mapping):
            return all(_finite(v) for v in value.values())
        if isinstance(value, list | tuple):
            return all(_finite(v) for v in value)
        return True

    if not _finite(diagnostics):
        _fail("B2_DLCM_AUXILIARY_DIAGNOSTICS_INVALID", "non-finite diagnostic value")

    payload = {
        "schema_version": "b2_dlcm_v2_auxiliary_diagnostics_v1",
        "diagnostic_source": "canonical_best_training_checkpoint",
        "not_available_from_deployment_artifact": True,
        "qualification_blocking": False,
        "diagnostics": dict(diagnostics),
    }
    payload["auxiliary_diagnostics_manifest_sha256"] = protocol.canonical_json_sha256(
        {k: v for k, v in payload.items() if k != "auxiliary_diagnostics_manifest_sha256"}
    )
    return payload


def reject_signed_proxy_from_deployment_weights(pred_values: Any) -> None:
    """Refuse allocation-weight proxies for signed diagnostics (reuse V1 check)."""

    if not isinstance(pred_values, torch.Tensor):
        return
    if v1_eval.looks_like_allocation_simplex(pred_values):
        _fail(
            "B2_DLCM_AUXILIARY_DIAGNOSTICS_INVALID",
            "deployment weights cannot proxy signed outputs",
        )


def build_final_decision_from_metrics(
    *,
    gt_target_learning: Mapping[str, Any],
    localization: Mapping[str, Any],
    gate_result: Mapping[str, Any],
) -> dict[str, Any]:
    verdict = "qualified" if gate_result.get("passed") else "unqualified"
    return protocol.build_h_decision(
        gt_target_learning=gt_target_learning,
        localization=localization,
        thresholds=dict(gate_result.get("thresholds", {})),
        verdict=verdict,
    )


def finalize_accepted_or_forbid(
    *,
    final_gate: Mapping[str, Any],
    h_deploy: str,
    h_decision: str,
    h_evidence: str,
    h_selection: str,
    upstream: Mapping[str, Any],
    v2_contract_sha256: str,
) -> dict[str, Any]:
    if final_gate.get("passed") is not True:
        _fail("B2_DLCM_ACCEPTED_MANIFEST_FORBIDDEN", "final failed; no H_accepted")
    return protocol.build_h_accepted(
        h_deploy=h_deploy,
        h_decision=h_decision,
        h_evidence=h_evidence,
        h_selection=h_selection,
        upstream=upstream,
        v2_contract_sha256=v2_contract_sha256,
        final_passed=True,
    )


__all__ = [
    "allocation_jsd",
    "build_auxiliary_diagnostics_manifest",
    "build_final_decision_from_metrics",
    "evaluate_development_gates",
    "finalize_accepted_or_forbid",
    "reject_signed_proxy_from_deployment_weights",
    "require_development_go",
    "spearman_average_ranks",
    "top1_set_agreement",
]

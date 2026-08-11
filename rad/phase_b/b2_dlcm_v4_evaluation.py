"""B2-05C3 V3 evaluation gates (thresholds identical to C1/V2)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NoReturn

from rad.phase_b import b2_dlcm_v4_protocol as protocol

GT_MACRO_MARGIN = 1e-5
GT_PER_CATEGORY_SLACK = 1e-4
LOC_MACRO_PIXEL_AP_MIN = 0.0
LOC_MACRO_PIXEL_AUROC_MIN = -1e-4
LOC_MACRO_AUPRO_MIN = -1e-4
LOC_PER_CATEGORY_FLOOR = -1e-3


class B2DLCMV4EvaluationError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMV4EvaluationError(code, detail)


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
    return {
        "passed": len(reasons) == 0,
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


def historical_comparison_diagnostic(
    *,
    v4_metrics: Mapping[str, float],
    c1_metrics: Mapping[str, float],
) -> dict[str, Any]:
    """Non-blocking C1/C2 comparison diagnostic (never a qualification gate)."""

    deltas = {
        key: float(v4_metrics.get(key, 0.0)) - float(c1_metrics.get(key, 0.0))
        for key in sorted(set(v4_metrics) | set(c1_metrics))
    }
    return {
        "qualification_blocking": False,
        "must_beat_c1": False,
        "deltas": deltas,
    }


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
        "schema_version": "b2_dlcm_v4_auxiliary_diagnostics_v1",
        "diagnostic_source": "canonical_best_training_checkpoint",
        "not_available_from_deployment_artifact": True,
        "qualification_blocking": False,
        "diagnostics": dict(diagnostics),
    }
    payload["auxiliary_diagnostics_manifest_sha256"] = protocol.canonical_json_sha256(
        {k: v for k, v in payload.items() if k != "auxiliary_diagnostics_manifest_sha256"}
    )
    return payload



def build_h_decision(
    *,
    final_gates: Mapping[str, Any],
    deployment_scientific_sha256: str,
    implementation_commit: str,
) -> dict[str, Any]:
    if final_gates.get("passed") is not True:
        _fail("B2_DLCM_DEVELOPMENT_UNQUALIFIED", "final gates must pass for H_decision")
    payload = {
        "schema_version": "b2_dlcm_v4_h_decision_v1",
        "final_gates_passed": True,
        "deployment_scientific_sha256": deployment_scientific_sha256,
        "implementation_commit": implementation_commit,
    }
    payload["h_decision_sha256"] = protocol.canonical_json_sha256(payload)
    return payload


c1_comparison_diagnostic = historical_comparison_diagnostic

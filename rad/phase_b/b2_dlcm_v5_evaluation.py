"""B2-05C4 V5 evaluation gates (thresholds identical to V4/V3/V2)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

GT_MACRO_MARGIN = 1e-5
GT_PER_CATEGORY_SLACK = 1e-4
LOC_MACRO_PIXEL_AP_MIN = 0.0
LOC_MACRO_PIXEL_AUROC_MIN = -1e-4
LOC_MACRO_AUPRO_MIN = -1e-4
LOC_PER_CATEGORY_FLOOR = -1e-3

C4_TERMINATION_CONCLUSION = (
    "current 18-D descriptor + 16-record training/calibration contract "
    "cannot stably satisfy the Carpet per-category GT target gate"
)


class B2DLCMV5EvaluationError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMV5EvaluationError(code, detail)


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


def c4_termination_payload(*, failed_reasons: Sequence[str]) -> dict[str, Any]:
    return {
        "development_verdict": "development_unqualified",
        "failed_reasons": list(failed_reasons),
        "enter_c5": False,
        "retune_loss_sampler_selector_beta_loo_forbidden": True,
        "final_remains_untouched": True,
        "lse_started": False,
        "conclusion": C4_TERMINATION_CONCLUSION,
        "next_step": (
            "descriptor sufficiency / training coverage / target variance / "
            "category-generalization protocol"
        ),
    }

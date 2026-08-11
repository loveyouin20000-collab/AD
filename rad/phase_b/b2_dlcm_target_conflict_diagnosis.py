"""B2-05C0 read-only teacher-fidelity / target-conflict diagnosis primitives.

No training, no checkpoint mutation, no accepted-manifest generation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from rad.phase_b import b2_dlcm_deployment as deployment
from rad.phase_b import b2_dlcm_evaluation as evaluation

FROZEN_QUALIFICATION_STATUS = "localized_but_target_fidelity_unqualified"
FROZEN_IDENTITIES = {
    "accepted_training_plan": "59e20f4cb337ef42384f70bb8b3dad5211d906341b0a2d41f7e6847610635980",
    "seed_collection": "94a6a9332a0694889c7a0255814ac13fe8316c601529197063165ce14ec1277f",
    "canonical_selection": "e3bc06dfa02d6109544648020680d907bf0fce5ed7a093372d74009f9e69e142",
    "deployment_scientific": "4cbc6fb88f39ed86deacfbbe48580f7682453b94becb046ec6ef1b1302df378a",
    "evaluation_unlock": "19dca41e9f647d12afce9877a7340f5af58bf9a23997d7339dded26d89fe73dd",
    "qualification_scientific": "da51e5fc1302cf507bc844f87e82cb66f7d2fa0a13e61f28a0dba14333201c49",
}

# Sentinel: diagnosis must never call into training. Tests monkeypatch this.
_FORBIDDEN_TRAINING_HOOK = None


class B2DLCMDiagnosisError(RuntimeError):
    """Hard fail-closed diagnosis guard violation."""


def kl_p_vs_w(p: torch.Tensor, w: torch.Tensor, *, eps: float = 1e-12) -> float:
    """D_KL(p || w) with simplex floor for diagnosis (softmax weights never hit exact 0).

    Exact zeros in positive-allocation targets make the evaluation helper return
    non-physical negative values when support(p) is not subset of support(w).
    Diagnosis therefore renormalizes a floored simplex; sealed qualification
    metrics continue to use ``b2_dlcm_evaluation._kl_p_vs_weights`` unchanged.
    """

    w64 = w.to(torch.float64).reshape(-1).clamp_min(float(eps))
    w64 = w64 / w64.sum()
    return evaluation._kl_p_vs_weights(p.to(torch.float32), w64.to(torch.float32))  # noqa: SLF001


def uniform_weights(n: int) -> torch.Tensor:
    return torch.full((n,), 1.0 / float(n), dtype=torch.float64)


def equal_family_oracle(p_gt: torch.Tensor, p_t: torch.Tensor) -> torch.Tensor:
    """Exact simplex optimum of 0.5 KL(p_gt||w) + 0.5 KL(p_t||w)."""

    w = 0.5 * p_gt.to(torch.float64) + 0.5 * p_t.to(torch.float64)
    s = float(w.sum().item())
    if s <= 0.0:
        raise B2DLCMDiagnosisError("equal_family_oracle: non-positive mass")
    return w / s


def weighted_family_oracle(p_gt: torch.Tensor, p_t: torch.Tensor, *, alpha: float) -> torch.Tensor:
    """Optimum of alpha KL(p_gt||w) + (1-alpha) KL(p_t||w) on the simplex."""

    if not (0.0 <= float(alpha) <= 1.0):
        raise B2DLCMDiagnosisError(f"alpha out of [0,1]: {alpha}")
    w = float(alpha) * p_gt.to(torch.float64) + (1.0 - float(alpha)) * p_t.to(torch.float64)
    s = float(w.sum().item())
    if s <= 0.0:
        raise B2DLCMDiagnosisError("weighted_family_oracle: non-positive mass")
    return w / s


def dual_family_mean_kl(p_gt: torch.Tensor, p_t: torch.Tensor, w: torch.Tensor) -> float:
    return 0.5 * kl_p_vs_w(p_gt, w) + 0.5 * kl_p_vs_w(p_t, w)


def alpha_grid() -> list[float]:
    return [round(i * 0.01, 10) for i in range(101)]


def oracle_candidate_metrics(
    *,
    p_gt: torch.Tensor,
    p_t: torch.Tensor,
    weights: torch.Tensor,
) -> dict[str, float]:
    u = uniform_weights(int(weights.numel()))
    kl_gt = kl_p_vs_w(p_gt, weights)
    kl_t = kl_p_vs_w(p_t, weights)
    kl_gt_u = kl_p_vs_w(p_gt, u)
    kl_t_u = kl_p_vs_w(p_t, u)
    return {
        "kl_gt": kl_gt,
        "kl_teacher": kl_t,
        "kl_gt_uniform": kl_gt_u,
        "kl_teacher_uniform": kl_t_u,
        "dual_family_mean_kl": 0.5 * kl_gt + 0.5 * kl_t,
        "gt_improvement_over_uniform": kl_gt_u - kl_gt,
        "teacher_improvement_over_uniform": kl_t_u - kl_t,
        "delta_from_uniform_gt": kl_gt - kl_gt_u,
        "delta_from_uniform_teacher": kl_t - kl_t_u,
    }


def empty_split_depth_report_scaffold() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("training", "calibration", "evaluation"):
        out[split] = {}
        for depth in (12, 18, 24):
            out[split][depth] = {
                "gt": {},
                "teacher": {},
                "equal_family_diagnostic_average": {},
            }
    return out


def finalize_alpha_feasibility_report(
    *,
    calibration: Mapping[str, Any],
    evaluation_posthoc: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "calibration_feasible_alphas": list(calibration.get("feasible_alphas", [])),
        "calibration_feasible_alpha_interval": list(calibration.get("interval", [])),
        "evaluation_posthoc_feasible_alphas": list(evaluation_posthoc.get("feasible_alphas", [])),
        "evaluation_posthoc_feasible_alpha_interval": list(evaluation_posthoc.get("interval", [])),
        "evaluation_used_for_alpha_selection": False,
        "note": "evaluation interval is diagnostic only; never used to choose alpha",
    }


def assert_checkpoint_bytes_immutable(path: Path | str, *, expected_bytes: bytes) -> None:
    observed = Path(path).read_bytes()
    if observed != expected_bytes:
        raise B2DLCMDiagnosisError("checkpoint immutable bytes mismatch (modified on disk)")


def guard_no_training_invocation(*, attempt_train: bool = False) -> None:
    if attempt_train:
        # Optional canary for tests; never proceed to training.
        if callable(_FORBIDDEN_TRAINING_HOOK):
            try:
                _FORBIDDEN_TRAINING_HOOK()
            except Exception:
                pass
        raise B2DLCMDiagnosisError("diagnosis forbids training invocation")


def guard_no_teacher_backbone_invocation(*, teacher_forward_count: int) -> None:
    if int(teacher_forward_count) != 0:
        raise B2DLCMDiagnosisError(
            f"diagnosis forbids teacher/backbone invocation (count={teacher_forward_count})"
        )


def guard_no_accepted_manifest_write(*, output_dir: Path | str, filename: str) -> None:
    name = Path(filename).name
    if "accepted" in name.lower() and "manifest" in name.lower():
        raise B2DLCMDiagnosisError(
            f"diagnosis forbids accepted-manifest generation: {name}"
        )
    path = Path(output_dir) / name
    if path.exists():
        raise B2DLCMDiagnosisError(f"accepted manifest unexpectedly present: {path}")


def assert_qualification_frozen(observed: Mapping[str, Any]) -> None:
    if observed.get("qualification_status") != FROZEN_QUALIFICATION_STATUS:
        raise B2DLCMDiagnosisError("qualification_status drifted")
    if observed.get("deployment_qualified") is not False:
        raise B2DLCMDiagnosisError("deployment_qualified drifted")
    if observed.get("accepted_deployment_manifest_created") is not False:
        raise B2DLCMDiagnosisError("accepted_deployment_manifest_created drifted")
    identities = observed.get("identities", {})
    if not isinstance(identities, Mapping):
        raise B2DLCMDiagnosisError("identities missing")
    for key, expected in FROZEN_IDENTITIES.items():
        if identities.get(key) != expected:
            raise B2DLCMDiagnosisError(f"identity drifted: {key}")


def refuse_signed_proxy_from_weights(weights: torch.Tensor, phi: torch.Tensor) -> float:
    """Diagnosis path must not use allocation weights as signed substitutes."""

    return evaluation.signed_huber_diagnostic(weights, phi)


def target_conflict_stats(
    p_gt: torch.Tensor,
    p_t: torch.Tensor,
    *,
    phi_gt: torch.Tensor | None = None,
    phi_t: torch.Tensor | None = None,
) -> dict[str, Any]:
    p_gt64 = p_gt.to(torch.float64).reshape(-1)
    p_t64 = p_t.to(torch.float64).reshape(-1)
    jsd = float(deployment.allocation_jsd(p_gt64.float(), p_t64.float()).item())
    l1 = float((p_gt64 - p_t64).abs().sum().item())
    l2 = float(torch.sqrt(((p_gt64 - p_t64) ** 2).sum()).item())
    top1 = deployment.top1_set_agreement(p_gt64.float(), p_t64.float())
    spearman = deployment.spearman_average_ranks(p_gt64.float(), p_t64.float())

    def _entropy(p: torch.Tensor) -> float:
        pos = p > 0
        return float((-(p[pos] * torch.log(p[pos]))).sum().item())

    support_gt = {i for i, v in enumerate(p_gt64.tolist()) if v > 1e-12}
    support_t = {i for i, v in enumerate(p_t64.tolist()) if v > 1e-12}
    overlap = len(support_gt & support_t) / float(max(1, len(support_gt | support_t)))
    out: dict[str, Any] = {
        "jsd_natural_log": jsd,
        "l1": l1,
        "l2": l2,
        "target_top1_agreement": top1,
        "target_spearman": spearman,
        "entropy_gt": _entropy(p_gt64),
        "entropy_teacher": _entropy(p_t64),
        "max_prob_gt": float(p_gt64.max().item()),
        "max_prob_teacher": float(p_t64.max().item()),
        "support_overlap": overlap,
        "teacher_flatter_than_gt": _entropy(p_t64) > _entropy(p_gt64),
        "teacher_more_uniform_than_gt": float(p_t64.max()) < float(p_gt64.max()),
    }
    if phi_gt is not None and phi_t is not None:
        out["signed_top1_agreement"] = deployment.top1_set_agreement(phi_gt.float(), phi_t.float())
        out["signed_spearman"] = deployment.spearman_average_ranks(phi_gt.float(), phi_t.float())
    return out


def family_gates_pass(
    *,
    kl_gt_macro: float,
    kl_gt_uniform_macro: float,
    kl_teacher_macro: float,
    kl_teacher_uniform_macro: float,
    per_category: Mapping[str, Mapping[str, float]],
) -> bool:
    """Mirror frozen V1 target-learning gates (macro + per-category)."""

    if not (
        kl_gt_macro <= kl_gt_uniform_macro - 1e-5
        and kl_teacher_macro <= kl_teacher_uniform_macro - 1e-5
    ):
        return False
    for fam in per_category.values():
        if fam["gt"] > fam["gt_uniform"] + 1e-4 or fam["teacher"] > fam["teacher_uniform"] + 1e-4:
            return False
    return True


def aggregate_category_macro(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
) -> dict[str, Any]:
    return deployment.aggregate_target_fidelity(list(rows), metric_path=(value_key,))


def compare_weights_to_targets(
    w: torch.Tensor,
    *,
    p_gt: torch.Tensor,
    p_t: torch.Tensor,
    equal_oracle: torch.Tensor,
) -> dict[str, Any]:
    u = uniform_weights(int(w.numel())).float()
    wf = w.float()
    return {
        "jsd_vs_gt": float(deployment.allocation_jsd(p_gt.float(), wf).item()),
        "jsd_vs_teacher": float(deployment.allocation_jsd(p_t.float(), wf).item()),
        "jsd_vs_equal_oracle": float(deployment.allocation_jsd(equal_oracle.float(), wf).item()),
        "jsd_vs_uniform": float(deployment.allocation_jsd(u, wf).item()),
        "l1_vs_gt": float((wf.double() - p_gt.double()).abs().sum().item()),
        "l1_vs_teacher": float((wf.double() - p_t.double()).abs().sum().item()),
        "l1_vs_equal_oracle": float((wf.double() - equal_oracle.double()).abs().sum().item()),
        "allocation_top1_vs_gt": deployment.top1_set_agreement(p_gt.float(), wf),
        "allocation_top1_vs_teacher": deployment.top1_set_agreement(p_t.float(), wf),
        "allocation_top1_vs_equal_oracle": deployment.top1_set_agreement(equal_oracle.float(), wf),
        "spearman_vs_gt": deployment.spearman_average_ranks(wf, p_gt.float()),
        "spearman_vs_teacher": deployment.spearman_average_ranks(wf, p_t.float()),
        "spearman_vs_equal_oracle": deployment.spearman_average_ranks(wf, equal_oracle.float()),
        "entropy_w": float((-(wf.double().clamp_min(1e-30) * torch.log(wf.double().clamp_min(1e-30)))).sum()),
        "max_weight": float(wf.max().item()),
    }


def diagnostic_selector_scores(
    *,
    kl_gt: float,
    kl_teacher: float,
    kl_gt_uniform: float,
    kl_teacher_uniform: float,
) -> dict[str, float | bool]:
    mean_family = 0.5 * kl_gt + 0.5 * kl_teacher
    worst_family = max(kl_gt, kl_teacher)
    worst_delta = max(kl_gt - kl_gt_uniform, kl_teacher - kl_teacher_uniform)
    improves = (kl_gt <= kl_gt_uniform - 1e-5) and (kl_teacher <= kl_teacher_uniform - 1e-5)
    return {
        "mean_family_kl": mean_family,
        "worst_family_kl": worst_family,
        "uniform_relative_worst_family_delta": worst_delta,
        "constrained_feasible": improves,
        "constrained_objective": mean_family if improves else math.inf,
    }


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def summarize_rows_by_category(
    rows: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str],
) -> dict[str, Any]:
    categories = sorted({str(r["category"]) for r in rows})
    out: dict[str, Any] = {"per_category": {}, "category_macro": {}, "pooled_sample_mean": {}, "per_sample": list(rows)}
    for key in keys:
        per_cat: dict[str, float] = {}
        for cat in categories:
            vals = [float(r[key]) for r in rows if str(r["category"]) == cat]
            per_cat[cat] = _mean(vals)
        out["per_category"][key] = per_cat
        out["category_macro"][key] = _mean([per_cat[c] for c in categories]) if categories else float("nan")
        out["pooled_sample_mean"][key] = _mean([float(r[key]) for r in rows])
    return out


def evaluate_weight_candidates_on_records(
    records: Sequence[Mapping[str, Any]],
    *,
    depth: int,
    weight_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> dict[str, Any]:
    """weight_fn(p_gt, p_t) -> weights tensor."""

    rows: list[dict[str, Any]] = []
    for record in records:
        p_gt = record["p_gt"][int(depth)].to(torch.float64)
        p_t = record["p_t"][int(depth)].to(torch.float64)
        w = weight_fn(p_gt, p_t).to(torch.float64)
        metrics = oracle_candidate_metrics(p_gt=p_gt, p_t=p_t, weights=w)
        rows.append(
            {
                "stable_sample_id": str(record["stable_sample_id"]),
                "category": str(record["category"]),
                **metrics,
            }
        )
    summary = summarize_rows_by_category(
        rows,
        keys=(
            "kl_gt",
            "kl_teacher",
            "kl_gt_uniform",
            "kl_teacher_uniform",
            "dual_family_mean_kl",
            "delta_from_uniform_gt",
            "delta_from_uniform_teacher",
            "gt_improvement_over_uniform",
            "teacher_improvement_over_uniform",
        ),
    )
    # Per-category maximum degradation vs uniform (positive = worse than uniform).
    max_deg: dict[str, dict[str, float]] = {"gt": {}, "teacher": {}}
    for cat, vals in summary["per_category"]["delta_from_uniform_gt"].items():
        max_deg["gt"][cat] = float(vals)
    for cat, vals in summary["per_category"]["delta_from_uniform_teacher"].items():
        max_deg["teacher"][cat] = float(vals)
    summary["per_category_max_degradation_vs_uniform"] = {
        "gt": max(max_deg["gt"].values()) if max_deg["gt"] else float("nan"),
        "teacher": max(max_deg["teacher"].values()) if max_deg["teacher"] else float("nan"),
        "per_category": max_deg,
    }
    gates_ok = family_gates_pass(
        kl_gt_macro=float(summary["category_macro"]["kl_gt"]),
        kl_gt_uniform_macro=float(summary["category_macro"]["kl_gt_uniform"]),
        kl_teacher_macro=float(summary["category_macro"]["kl_teacher"]),
        kl_teacher_uniform_macro=float(summary["category_macro"]["kl_teacher_uniform"]),
        per_category={
            cat: {
                "gt": float(summary["per_category"]["kl_gt"][cat]),
                "gt_uniform": float(summary["per_category"]["kl_gt_uniform"][cat]),
                "teacher": float(summary["per_category"]["kl_teacher"][cat]),
                "teacher_uniform": float(summary["per_category"]["kl_teacher_uniform"][cat]),
            }
            for cat in summary["per_category"]["kl_gt"]
        },
    )
    summary["both_target_learning_gates_pass"] = gates_ok
    return summary


def alpha_feasibility_on_records(
    records: Sequence[Mapping[str, Any]],
    *,
    depth: int = 24,
) -> dict[str, Any]:
    feasible: list[float] = []
    per_alpha: list[dict[str, Any]] = []
    for alpha in alpha_grid():

        def _weight_fn(
            p_gt: torch.Tensor,
            p_t: torch.Tensor,
            *,
            _alpha: float = float(alpha),
        ) -> torch.Tensor:
            return weighted_family_oracle(p_gt, p_t, alpha=_alpha)

        summary = evaluate_weight_candidates_on_records(
            records,
            depth=depth,
            weight_fn=_weight_fn,
        )
        entry = {
            "alpha": alpha,
            "kl_gt_macro": summary["category_macro"]["kl_gt"],
            "kl_teacher_macro": summary["category_macro"]["kl_teacher"],
            "gt_max_degradation": summary["per_category_max_degradation_vs_uniform"]["gt"],
            "teacher_max_degradation": summary["per_category_max_degradation_vs_uniform"]["teacher"],
            "both_gates_pass": summary["both_target_learning_gates_pass"],
        }
        per_alpha.append(entry)
        if entry["both_gates_pass"]:
            feasible.append(alpha)
    interval = [feasible[0], feasible[-1]] if feasible else []
    return {"feasible_alphas": feasible, "interval": interval, "per_alpha": per_alpha}


def predict_model_weights_on_record(
    model: Any,
    record: Mapping[str, Any],
    *,
    depth: int,
    candidate_layers: Sequence[int],
) -> torch.Tensor:
    from rad.phase_b import b2_dlcm as dlcm
    from rad.phase_b import b2_dlcm_evaluation as evaluation_mod

    players = dlcm.players_for_depth(candidate_layers, depth)
    desc = record["descriptors"][int(depth)]
    if desc.ndim == 3:
        desc = desc.reshape(desc.shape[-2], desc.shape[-1])
    _logits, weights = evaluation_mod.predict_weights(
        model, desc, prediction_depth=int(depth), player_layer_ids=players
    )
    return weights.to(torch.float64)

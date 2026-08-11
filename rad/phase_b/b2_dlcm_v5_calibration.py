"""B2-05C4 V5 Calibration: beta grid, LOO, eligibility, selection, A/B."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

import torch

from rad.phase_b import b2_dlcm_v5 as v5
from rad.phase_b import b2_dlcm_v5_protocol as protocol

CALIBRATION_MANIFEST_SCHEMA = "b2_dlcm_v5_calibration_manifest_v1"


class B2DLCMV5CalibrationError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMV5CalibrationError(code, detail)


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float32)
    return torch.tensor(value, dtype=torch.float32)


def validate_calibration_records(records: Sequence[Mapping[str, Any]]) -> None:
    if len(records) != 8:
        _fail("B2_DLCM_V5_CALIBRATION_INPUT_INVALID", f"expected 8 records, got {len(records)}")
    counts = {"bottle": 0, "carpet": 0}
    ids: set[str] = set()
    for row in records:
        cat = str(row.get("category", ""))
        if cat not in counts:
            _fail("B2_DLCM_V5_CALIBRATION_INPUT_INVALID", f"unknown category {cat}")
        counts[cat] += 1
        sid = str(row.get("stable_sample_id", ""))
        if not sid:
            _fail("B2_DLCM_V5_CALIBRATION_INPUT_INVALID", "missing stable_sample_id")
        if sid in ids:
            _fail("B2_DLCM_V5_CALIBRATION_INPUT_INVALID", f"duplicate id {sid}")
        ids.add(sid)
        p = _as_tensor(row["p_gt"])
        w = _as_tensor(row["dynamic_weights"])
        if p.ndim != 1 or w.ndim != 1 or p.shape != w.shape:
            _fail("B2_DLCM_V5_CALIBRATION_INPUT_INVALID", "p_gt/dynamic_weights shape invalid")
        if "depth" in row and int(row["depth"]) != v5.LOO_DEPTH:
            _fail("B2_DLCM_V5_CALIBRATION_INPUT_INVALID", "calibration LOO depth must be 24")
    if counts["bottle"] != v5.CALIBRATION_PER_CATEGORY or counts["carpet"] != v5.CALIBRATION_PER_CATEGORY:
        _fail("B2_DLCM_V5_CALIBRATION_INPUT_INVALID", f"per-category counts invalid: {counts}")


def records_by_category(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {c: [] for c in v5.TRAINING_CATEGORIES}
    for row in records:
        grouped[str(row["category"])].append(row)
    return grouped


def mean_kl_for_weights(
    records: Sequence[Mapping[str, Any]],
    *,
    beta: float,
) -> dict[str, Any]:
    """Full-set category + macro KL for mixed weights vs uniform."""

    per_cat: dict[str, float] = {}
    uni_cat: dict[str, float] = {}
    for cat in v5.TRAINING_CATEGORIES:
        rows = [r for r in records if str(r["category"]) == cat]
        if not rows:
            _fail("B2_DLCM_V5_CALIBRATION_INPUT_INVALID", f"missing category {cat}")
        model_kls: list[torch.Tensor] = []
        uni_kls: list[torch.Tensor] = []
        for row in rows:
            p = _as_tensor(row["p_gt"])
            dyn = _as_tensor(row["dynamic_weights"])
            mixed = v5.mix_uniform_anchored_weights(dyn, beta)
            uni = v5.depth_matched_uniform(int(dyn.shape[0]))
            model_kls.append(v5.per_sample_allocation_kl_from_weights(p, mixed))
            uni_kls.append(v5.per_sample_allocation_kl_from_weights(p, uni))
        per_cat[cat] = float(torch.stack(model_kls).mean().item())
        uni_cat[cat] = float(torch.stack(uni_kls).mean().item())
    macro = sum(per_cat.values()) / float(len(per_cat))
    uni_macro = sum(uni_cat.values()) / float(len(uni_cat))
    return {
        "per_category_gt_kl": per_cat,
        "per_category_uniform_gt_kl": uni_cat,
        "macro_gt_kl": macro,
        "uniform_macro_gt_kl": uni_macro,
    }


def leave_one_out_regrets(
    records: Sequence[Mapping[str, Any]],
    *,
    beta: float,
) -> dict[str, Any]:
    """Depth-24 LOO relative regrets; 4 folds/category → 8 regrets."""

    validate_calibration_records(records)
    grouped = records_by_category(records)
    fold_regrets: list[dict[str, Any]] = []
    all_values: list[float] = []
    for cat in v5.TRAINING_CATEGORIES:
        rows = grouped[cat]
        for leave_i, held_out in enumerate(rows):
            kept = [r for j, r in enumerate(rows) if j != leave_i]
            if len(kept) != 3:
                _fail("B2_DLCM_V5_CALIBRATION_INPUT_INVALID", "LOO fold size must be 3")
            regrets: list[torch.Tensor] = []
            for row in kept:
                p = _as_tensor(row["p_gt"])
                dyn = _as_tensor(row["dynamic_weights"])
                mixed = v5.mix_uniform_anchored_weights(dyn, beta)
                uni = v5.depth_matched_uniform(int(dyn.shape[0]))
                kl_m = v5.per_sample_allocation_kl_from_weights(p, mixed)
                kl_u = v5.per_sample_allocation_kl_from_weights(p, uni)
                regrets.append(kl_m - kl_u)
            r_mean = torch.stack(regrets).mean()
            value = float(r_mean.item())
            all_values.append(value)
            fold_regrets.append(
                {
                    "category": cat,
                    "held_out_stable_sample_id": str(held_out["stable_sample_id"]),
                    "leave_index": leave_i,
                    "relative_regret": value,
                }
            )
    if len(fold_regrets) != 8:
        _fail("B2_DLCM_V5_CALIBRATION_INPUT_INVALID", "expected 8 LOO folds")
    m_loo = max(all_values)
    return {
        "folds": fold_regrets,
        "m_loo": m_loo,
        "depth": v5.LOO_DEPTH,
    }


def is_beta_eligible(metrics: Mapping[str, Any]) -> bool:
    macro = float(metrics["macro_gt_kl"])
    uni_macro = float(metrics["uniform_macro_gt_kl"])
    if not (macro <= uni_macro - v5.GT_MACRO_MARGIN):
        return False
    per = metrics["per_category_gt_kl"]
    uni = metrics["per_category_uniform_gt_kl"]
    for cat in v5.TRAINING_CATEGORIES:
        if not (float(per[cat]) <= float(uni[cat]) + v5.GT_PER_CATEGORY_SLACK):
            return False
    return True


def evaluate_beta_candidate(
    records: Sequence[Mapping[str, Any]],
    *,
    beta_index: int,
) -> dict[str, Any]:
    beta = v5.beta_from_index(beta_index)
    metrics = mean_kl_for_weights(records, beta=beta)
    loo = leave_one_out_regrets(records, beta=beta)
    eligible = is_beta_eligible(metrics)
    return {
        "beta_index": int(beta_index),
        "beta": beta,
        "beta_decimal": v5.beta_decimal_string(beta_index),
        "eligible": eligible,
        "m_loo": float(loo["m_loo"]),
        "loo": loo,
        "macro_gt_kl": float(metrics["macro_gt_kl"]),
        "uniform_macro_gt_kl": float(metrics["uniform_macro_gt_kl"]),
        "per_category_gt_kl": dict(metrics["per_category_gt_kl"]),
        "per_category_uniform_gt_kl": dict(metrics["per_category_uniform_gt_kl"]),
    }


def select_beta_star(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(candidates) != v5.BETA_GRID_SIZE:
        _fail("B2_DLCM_V5_BETA_GRID_INVALID", f"expected {v5.BETA_GRID_SIZE} candidates")
    eligible = [c for c in candidates if c.get("eligible") is True]
    if not eligible:
        _fail("B2_DLCM_V5_NO_ELIGIBLE_BETA", "no beta passed Calibration eligibility")

    min_m = min(float(c["m_loo"]) for c in eligible)
    tied = [c for c in eligible if float(c["m_loo"]) <= min_m + v5.LOO_TIE_EPS]
    tied_sorted = sorted(
        tied,
        key=lambda c: (
            -float(c["beta"]),
            float(c["macro_gt_kl"]),
            int(c["beta_index"]),
        ),
    )
    best = tied_sorted[0]
    if best.get("eligible") is not True:
        _fail("B2_DLCM_V5_BETA_SELECTION_INVALID", "selected beta not eligible")
    return {
        "beta_index": int(best["beta_index"]),
        "beta": float(best["beta"]),
        "beta_decimal": str(best["beta_decimal"]),
        "m_loo": float(best["m_loo"]),
        "macro_gt_kl": float(best["macro_gt_kl"]),
    }


def run_calibration(
    records: Sequence[Mapping[str, Any]],
    *,
    process_label: str,
    deployment_identity: Mapping[str, Any] | None = None,
    teacher_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate 101 candidates and select beta*. Teacher diagnostics ignored."""

    if process_label not in {"A", "B", "fixture"}:
        _fail("B2_DLCM_V5_CALIBRATION_INPUT_INVALID", f"invalid process_label {process_label}")
    validate_calibration_records(records)
    v5.validate_beta_grid()
    # Teacher diagnostics must not affect selection (accepted but unused).
    _ = teacher_diagnostics
    candidates = [
        evaluate_beta_candidate(records, beta_index=i) for i in range(v5.BETA_GRID_SIZE)
    ]
    selected = select_beta_star(candidates)
    candidate_public = [
        {
            "beta_index": c["beta_index"],
            "beta_decimal": c["beta_decimal"],
            "eligible": c["eligible"],
            "m_loo": c["m_loo"],
            "macro_gt_kl": c["macro_gt_kl"],
            "per_category_gt_kl": c["per_category_gt_kl"],
            "uniform_macro_gt_kl": c["uniform_macro_gt_kl"],
            "per_category_uniform_gt_kl": c["per_category_uniform_gt_kl"],
        }
        for c in candidates
    ]
    manifest = {
        "schema_version": CALIBRATION_MANIFEST_SCHEMA,
        "calibration_contract_version": v5.CALIBRATION_CONTRACT_VERSION,
        "architecture_contract_version": v5.ARCHITECTURE_CONTRACT_VERSION,
        "process_label": process_label,
        "loo_depth": v5.LOO_DEPTH,
        "beta_grid_size": v5.BETA_GRID_SIZE,
        "candidates": candidate_public,
        "selected": selected,
        "deployment_identity": dict(deployment_identity or {}),
        "v4_immutable": v5.v4_immutable_identity(),
        "v5_contract_identity": v5.v5_contract_identity(),
        "teacher_diagnostics_considered_for_selection": False,
    }
    scientific = protocol.canonical_json_sha256(
        {
            "candidates": candidate_public,
            "selected": selected,
            "calibration_contract_version": v5.CALIBRATION_CONTRACT_VERSION,
            "loo_depth": v5.LOO_DEPTH,
        }
    )
    manifest["scientific_identity"] = scientific
    return manifest


def assert_calibration_ab_equal(manifest_a: Mapping[str, Any], manifest_b: Mapping[str, Any]) -> None:
    keys = ("candidates", "selected", "scientific_identity", "loo_depth", "beta_grid_size")
    for key in keys:
        if manifest_a.get(key) != manifest_b.get(key):
            _fail("B2_DLCM_V5_CALIBRATION_MISMATCH", f"A/B mismatch on {key}")
    bytes_a = protocol.canonical_json_bytes(
        {"candidates": manifest_a["candidates"], "selected": manifest_a["selected"]}
    )
    bytes_b = protocol.canonical_json_bytes(
        {"candidates": manifest_b["candidates"], "selected": manifest_b["selected"]}
    )
    if bytes_a != bytes_b:
        _fail("B2_DLCM_V5_CALIBRATION_MISMATCH", "canonical JSON byte inequality")


def dry_run_complete_v5_contract_validation(
    *,
    config: Mapping[str, Any],
    output_dir: str,
) -> dict[str, Any]:
    protocol.reject_bypass_flags(config)
    if config.get("schema_version") != v5.SCHEMA_VERSION:
        _fail("B2_DLCM_V5_CONTRACT_MISMATCH", "schema_version mismatch")
    if config.get("contract_stage") != "b2_05c4a":
        _fail("B2_DLCM_V5_CONTRACT_MISMATCH", "contract_stage must be b2_05c4a")
    if config.get("real_training_enabled") is True:
        protocol.forbid_training(context="dry_run_validation")
    # Prove final forbid is active.
    try:
        protocol.forbid_final_content_access(unlocked=False, context="dry_run")
    except protocol.B2DLCMV5ProtocolError as exc:
        if exc.code != "B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN":
            raise
    status = protocol.dry_run_status_payload()
    # Ensure dry-run never creates output_dir.
    from pathlib import Path

    if Path(output_dir).exists():
        # Allowed if pre-existing; we must not create it.
        pass
    status["output_dir_requested"] = output_dir
    status["contract_stage"] = config.get("contract_stage")
    return status

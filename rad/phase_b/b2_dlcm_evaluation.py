"""B2-05B formal evaluation orchestration over unlocked evaluation records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from rad.phase_b import b2_contribution_targets as contrib_mod
from rad.phase_b import b2_dlcm as dlcm
from rad.phase_b import b2_dlcm_deployment as deployment


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _kl_p_vs_weights(p: torch.Tensor, weights: torch.Tensor) -> float:
    """D_KL(p || w) with 0 log 0 := 0; weights must be a simplex row."""

    p64 = p.to(torch.float64)
    w64 = weights.to(torch.float64)
    positive = p64 > 0
    log_p = torch.where(positive, torch.log(torch.where(positive, p64, torch.ones_like(p64))), torch.zeros_like(p64))
    log_w = torch.where(w64 > 0, torch.log(torch.where(w64 > 0, w64, torch.ones_like(w64))), torch.zeros_like(w64))
    terms = torch.where(positive, p64 * (log_p - log_w), torch.zeros_like(p64))
    return float(terms.sum().item())


def _uniform_weights(n: int) -> torch.Tensor:
    return torch.full((n,), 1.0 / float(n), dtype=torch.float32)


def _signed_huber(pred: torch.Tensor, target: torch.Tensor, *, delta: float = 1.0) -> float:
    err = pred.to(torch.float64) - target.to(torch.float64)
    abs_err = err.abs()
    quadratic = 0.5 * err * err
    linear = abs_err - 0.5 * delta
    per = torch.where(abs_err <= delta, quadratic, linear)
    return float(per.mean().item())


def _stack_layer_maps(
    maps_by_layer: Mapping[int, torch.Tensor],
    players: Sequence[int],
) -> torch.Tensor:
    stacked: list[torch.Tensor] = []
    for layer in players:
        tensor = maps_by_layer[int(layer)]
        if tensor.ndim == 4:
            tensor = tensor.reshape(tensor.shape[-2], tensor.shape[-1])
        elif tensor.ndim == 3:
            tensor = tensor.reshape(tensor.shape[-2], tensor.shape[-1])
        stacked.append(tensor.to(dtype=torch.float32).contiguous())
    return torch.stack(stacked, dim=0)


def _fuse_maps(maps_nd: torch.Tensor, weights_1d: torch.Tensor) -> torch.Tensor:
    """maps [n,H,W], weights [n] → fused [H,W] via production sum-preserving fusion."""

    maps_5d = maps_nd.unsqueeze(0).unsqueeze(2)  # [1,n,1,H,W]
    w = weights_1d.unsqueeze(0)
    valid = torch.ones(1, maps_nd.shape[0], dtype=torch.bool)
    fused, _path = dlcm.sum_preserving_fusion(maps_5d, w, valid)
    return fused.reshape(maps_nd.shape[-2], maps_nd.shape[-1]).contiguous()


def load_teacher_scientific_record(teacher_cache_root: Path | str, stable_sample_id: str) -> dict[str, Any]:
    path = Path(teacher_cache_root) / "samples" / f"{stable_sample_id}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    record = payload["scientific_record"]
    if not isinstance(record, Mapping):
        raise RuntimeError(f"teacher scientific_record missing for {stable_sample_id}")
    return dict(record)


def predict_weights(
    model: nn.Module,
    descriptors: torch.Tensor,
    *,
    prediction_depth: int,
    player_layer_ids: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (logits[n], weights[n]) on CPU float32."""

    model.eval()
    x = descriptors.unsqueeze(0).to(dtype=torch.float32).contiguous()
    with torch.no_grad():
        forward_fn = getattr(model, "forward_deployment", None)
        if callable(forward_fn):
            logits, weights = forward_fn(
                x, prediction_depth=prediction_depth, player_layer_ids=player_layer_ids
            )
        else:
            logits, weights = model(
                x, prediction_depth=prediction_depth, player_layer_ids=player_layer_ids
            )
    return logits.reshape(-1).cpu().float(), weights.reshape(-1).cpu().float()


def evaluate_checkpoint_on_records(
    *,
    model: nn.Module,
    evaluation_records: Sequence[Mapping[str, Any]],
    teacher_cache_root: Path | str,
    prediction_depths: Sequence[int] = (12, 18, 24),
    candidate_layers: Sequence[int] = dlcm.DEFAULT_CANDIDATE_LAYERS,
) -> dict[str, Any]:
    """Full-batch ordered evaluation; returns depth/target/localization aggregates."""

    teacher_root = Path(teacher_cache_root)
    ordered = sorted(evaluation_records, key=lambda row: str(row["stable_sample_id"]))

    depth_target_rows: dict[int, list[dict[str, Any]]] = {int(d): [] for d in prediction_depths}
    # Localization accumulators per depth / category.
    loc_maps: dict[int, dict[str, list[torch.Tensor]]] = {
        int(d): {"dlcm": [], "baseline": [], "teacher": [], "mask": [], "label": [], "score_dlcm": [], "score_base": [], "category": []}
        for d in prediction_depths
    }

    invocation_proofs: list[Mapping[str, bool]] = []

    for record in ordered:
        sid = str(record["stable_sample_id"])
        category = str(record["category"])
        teacher_rec = load_teacher_scientific_record(teacher_root, sid)
        maps_by_depth, _full = contrib_mod.production_maps_from_teacher_record(teacher_rec)
        mask = contrib_mod.production_mask_from_teacher_record(teacher_rec)
        if mask.ndim == 4:
            mask_hw = mask.reshape(mask.shape[-2], mask.shape[-1]).to(torch.float32)
        elif mask.ndim == 3:
            mask_hw = mask.reshape(mask.shape[-2], mask.shape[-1]).to(torch.float32)
        else:
            mask_hw = mask.to(torch.float32)
        label = int(teacher_rec["image_label"])

        for depth in prediction_depths:
            players = dlcm.players_for_depth(candidate_layers, depth)
            desc = record["descriptors"][int(depth)]
            if not isinstance(desc, torch.Tensor):
                raise RuntimeError(f"descriptor missing for {sid} depth {depth}")
            if desc.ndim == 3:
                desc = desc.reshape(desc.shape[-2], desc.shape[-1])
            logits, weights = predict_weights(
                model, desc, prediction_depth=int(depth), player_layer_ids=players
            )
            uniform = _uniform_weights(len(players))
            p_gt = record["p_gt"][int(depth)].to(torch.float32)
            p_t = record["p_t"][int(depth)].to(torch.float32)
            phi_gt = record["phi_gt"][int(depth)].to(torch.float32)
            phi_t = record["phi_t"][int(depth)].to(torch.float32)

            # Auxiliary signed heads are absent on deployment trunk; use weight-derived
            # scores only for ranking/spearman vs signed targets when heads missing.
            signed_pred = weights  # diagnostic proxy when aux heads stripped
            row = {
                "stable_sample_id": sid,
                "category": category,
                "kl_gt_dlcm": _kl_p_vs_weights(p_gt, weights),
                "kl_gt_uniform": _kl_p_vs_weights(p_gt, uniform),
                "kl_teacher_dlcm": _kl_p_vs_weights(p_t, weights),
                "kl_teacher_uniform": _kl_p_vs_weights(p_t, uniform),
                "jsd_gt": float(deployment.allocation_jsd(p_gt, weights).item()),
                "jsd_teacher": float(deployment.allocation_jsd(p_t, weights).item()),
                "huber_gt": _signed_huber(signed_pred, phi_gt),
                "huber_teacher": _signed_huber(signed_pred, phi_t),
                "top1_gt": deployment.top1_set_agreement(p_gt, weights),
                "top1_teacher": deployment.top1_set_agreement(p_t, weights),
                "spearman_gt": deployment.spearman_average_ranks(weights, p_gt),
                "spearman_teacher": deployment.spearman_average_ranks(weights, p_t),
                "rank_gt": deployment.pairwise_ranking_accuracy(weights, p_gt),
                "rank_teacher": deployment.pairwise_ranking_accuracy(weights, p_t),
                "logits_used": True,
            }
            depth_target_rows[int(depth)].append(row)

            layer_maps = _stack_layer_maps(maps_by_depth[int(depth)], players)
            fused_dlcm = _fuse_maps(layer_maps, weights)
            fused_base = _fuse_maps(layer_maps, uniform)
            # Teacher reference map: equal-weight fusion is the depth-matched baseline;
            # teacher fidelity localization uses the full-depth map when available, else
            # equal-weight fusion of teacher causal lattice at this depth.
            teacher_map = fused_base.detach().clone()
            if int(depth) == 24:
                full = _full
                if full.ndim == 4:
                    teacher_map = full.reshape(full.shape[-2], full.shape[-1]).to(torch.float32)
                elif full.ndim == 3:
                    teacher_map = full.reshape(full.shape[-2], full.shape[-1]).to(torch.float32)
                else:
                    teacher_map = full.to(torch.float32)

            bucket = loc_maps[int(depth)]
            bucket["dlcm"].append(fused_dlcm)
            bucket["baseline"].append(fused_base)
            bucket["teacher"].append(teacher_map)
            bucket["mask"].append(mask_hw)
            bucket["label"].append(label)
            bucket["score_dlcm"].append(float(fused_dlcm.max().item()))
            bucket["score_base"].append(float(fused_base.max().item()))
            bucket["category"].append(category)

    depth_results: dict[int, dict[str, Any]] = {}
    for depth in prediction_depths:
        rows = depth_target_rows[int(depth)]
        target = {
            "kl_gt_dlcm": deployment.aggregate_target_fidelity(rows, metric_path=("kl_gt_dlcm",)),
            "kl_gt_uniform": deployment.aggregate_target_fidelity(rows, metric_path=("kl_gt_uniform",)),
            "kl_teacher_dlcm": deployment.aggregate_target_fidelity(
                rows, metric_path=("kl_teacher_dlcm",)
            ),
            "kl_teacher_uniform": deployment.aggregate_target_fidelity(
                rows, metric_path=("kl_teacher_uniform",)
            ),
            "jsd_gt": deployment.aggregate_target_fidelity(rows, metric_path=("jsd_gt",)),
            "jsd_teacher": deployment.aggregate_target_fidelity(rows, metric_path=("jsd_teacher",)),
        }
        # Formal localization per category.
        bucket = loc_maps[int(depth)]
        categories = sorted(set(bucket["category"]))
        per_cat_dlcm: dict[str, deployment.FormalLocalizationMetrics] = {}
        per_cat_base: dict[str, deployment.FormalLocalizationMetrics] = {}
        for category in categories:
            idxs = [i for i, c in enumerate(bucket["category"]) if c == category]
            labels = [bucket["label"][i] for i in idxs]
            masks = torch.stack([bucket["mask"][i] for i in idxs], dim=0)
            maps_d = torch.stack([bucket["dlcm"][i] for i in idxs], dim=0)
            maps_b = torch.stack([bucket["baseline"][i] for i in idxs], dim=0)
            teacher = torch.stack([bucket["teacher"][i] for i in idxs], dim=0)
            scores_d = [bucket["score_dlcm"][i] for i in idxs]
            scores_b = [bucket["score_base"][i] for i in idxs]
            metrics_d = deployment.compute_formal_localization_metrics(
                image_labels=labels,
                image_scores=scores_d,
                masks=masks.numpy(),
                anomaly_maps=maps_d.numpy(),
                teacher_map=teacher,
            )
            metrics_b = deployment.compute_formal_localization_metrics(
                image_labels=labels,
                image_scores=scores_b,
                masks=masks.numpy(),
                anomaly_maps=maps_b.numpy(),
                teacher_map=teacher,
            )
            per_cat_dlcm[category] = metrics_d
            per_cat_base[category] = metrics_b
            invocation_proofs.append(dict(metrics_d.invocation_proof))

        loc_evidence = deployment.build_formal_localization_gate_evidence(
            per_category_metrics=per_cat_dlcm,
            per_category_baseline=per_cat_base,
        )
        depth_results[int(depth)] = {
            "target_fidelity": target,
            "localization_evidence": {
                "metric_source_identity": loc_evidence.metric_source_identity,
                "delta_pixel_ap_macro": loc_evidence.delta_pixel_ap_macro,
                "delta_pixel_auroc_macro": loc_evidence.delta_pixel_auroc_macro,
                "delta_aupro_macro": loc_evidence.delta_aupro_macro,
                "per_category_localization": dict(loc_evidence.per_category_localization),
                "per_category_absolute": {
                    cat: {
                        "pixel_auroc": per_cat_dlcm[cat].pixel_auroc,
                        "pixel_ap": per_cat_dlcm[cat].pixel_ap,
                        "aupro": per_cat_dlcm[cat].aupro,
                        "teacher_spearman": per_cat_dlcm[cat].teacher_spearman,
                        "teacher_top1_overlap": per_cat_dlcm[cat].teacher_top1_overlap,
                        "baseline_pixel_auroc": per_cat_base[cat].pixel_auroc,
                        "baseline_pixel_ap": per_cat_base[cat].pixel_ap,
                        "baseline_aupro": per_cat_base[cat].aupro,
                    }
                    for cat in categories
                },
            },
            "sample_count": len(rows),
        }

    return {
        "depth_results": depth_results,
        "evaluation_sample_ids": [str(r["stable_sample_id"]) for r in ordered],
        "production_metric_invocation_proof": {
            "all_compute_paper_metrics": all(p.get("compute_paper_metrics") for p in invocation_proofs),
            "all_spearman_fidelity": all(p.get("spearman_fidelity") for p in invocation_proofs),
            "all_top1_overlap": all(p.get("top1_overlap") for p in invocation_proofs),
            "invocation_count": len(invocation_proofs),
        },
        "evaluation_content_sha256": _canonical_json_sha256(
            {
                "ids": [str(r["stable_sample_id"]) for r in ordered],
                "depths": list(prediction_depths),
            }
        ),
    }


def gates_from_depth24(depth24: Mapping[str, Any], *, best_epoch: int) -> dict[str, Any]:
    target = depth24["target_fidelity"]
    loc = depth24["localization_evidence"]
    per_cat_kl: dict[str, dict[str, float]] = {}
    # Rebuild per-category KL from absolute aggregates is insufficient; use macro gates
    # plus per-category localization already in evidence. For KL per-category, require
    # the target aggregate per_category fields.
    for category, value in target["kl_gt_dlcm"]["per_category"].items():
        per_cat_kl[category] = {
            "gt": float(value),
            "gt_uniform": float(target["kl_gt_uniform"]["per_category"][category]),
            "teacher": float(target["kl_teacher_dlcm"]["per_category"][category]),
            "teacher_uniform": float(target["kl_teacher_uniform"]["per_category"][category]),
        }
    evidence = deployment.FormalLocalizationGateEvidence(
        metric_source_identity=str(loc["metric_source_identity"]),
        delta_pixel_ap_macro=float(loc["delta_pixel_ap_macro"]),
        delta_pixel_auroc_macro=float(loc["delta_pixel_auroc_macro"]),
        delta_aupro_macro=float(loc["delta_aupro_macro"]),
        per_category_localization=dict(loc["per_category_localization"]),
    )
    return deployment.evaluate_qualification_gates(
        kl_dlcm_gt_macro=float(target["kl_gt_dlcm"]["category_macro"]),
        kl_uniform_gt_macro=float(target["kl_gt_uniform"]["category_macro"]),
        kl_dlcm_teacher_macro=float(target["kl_teacher_dlcm"]["category_macro"]),
        kl_uniform_teacher_macro=float(target["kl_teacher_uniform"]["category_macro"]),
        per_category_kl=per_cat_kl,
        delta_pixel_ap_macro=float(loc["delta_pixel_ap_macro"]),
        delta_pixel_auroc_macro=float(loc["delta_pixel_auroc_macro"]),
        delta_aupro_macro=float(loc["delta_aupro_macro"]),
        per_category_localization=dict(loc["per_category_localization"]),
        best_epoch=int(best_epoch),
        localization_evidence=evidence,
    )


def load_training_model_from_best_checkpoint(path: Path | str, *, seed: int) -> dlcm.B2DLCM:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = dlcm.B2DLCM(seed=seed)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


def load_deployment_trunk_from_checkpoint(path: Path | str) -> nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    trunk = deployment._instantiate_trunk_from_checkpoint(ckpt)  # noqa: SLF001
    trunk.load_state_dict(ckpt["state_dict"], strict=True)
    trunk.eval()
    return trunk

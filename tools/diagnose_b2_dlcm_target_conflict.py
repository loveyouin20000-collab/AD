#!/usr/bin/env python3
"""B2-05C0 read-only diagnosis of DLCM GT–teacher target conflict.

Loads verified artifacts only. No training, no checkpoint mutation,
no accepted-manifest generation, no teacher/backbone rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEFAULT_RUN = (
    _REPO_ROOT
    / "artifacts/phase_b/b2_dlcm_training/authoritative-run-20260804-152248"
)
DEFAULT_DESC = Path(
    "/root/autodl-tmp/AD-phase-b2-descriptor-real-extraction/"
    "artifacts/phase_b/b2_descriptor_artifacts/authoritative-run-a-20260729-013956"
)
DEFAULT_CONTRIB = Path(
    "/root/autodl-tmp/AD-phase-b2-contribution-target-materialization/"
    "artifacts/phase_b/b2_contribution_targets/authoritative-run-a-20260804-030431"
)
DEFAULT_TEACHER = Path(
    "/root/autodl-tmp/AD-phase-b2-teacher-cache-gpu/"
    "artifacts/phase_b/b2_teacher_cache/authoritative-run-a-20260723-155404"
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def _exact_float(value: Any) -> float:
    if isinstance(value, dict) and "value" in value:
        return float(value["value"])
    return float(value)


def _load_trace_epochs(trace_path: Path, epochs: list[int]) -> dict[int, dict[str, float]]:
    nodes = json.loads(trace_path.read_text(encoding="utf-8"))["nodes"]
    by_epoch = {int(n["record"]["epoch"]): n["record"] for n in nodes}
    out: dict[int, dict[str, float]] = {}
    for ep in epochs:
        if ep not in by_epoch:
            continue
        rec = by_epoch[ep]
        out[ep] = {
            "epoch": ep,
            "calibration_primary": _exact_float(rec["calibration_primary"]),
            "calibration_secondary": _exact_float(rec["calibration_secondary"]),
            "training_total_loss": _exact_float(rec.get("training_total_loss", float("nan"))),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--descriptor-root", type=Path, default=DEFAULT_DESC)
    parser.add_argument("--contribution-target-root", type=Path, default=DEFAULT_CONTRIB)
    parser.add_argument("--teacher-cache-root", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import torch

    from rad.phase_b import b2_dlcm as dlcm
    from rad.phase_b import b2_dlcm_evaluation as evaluation
    from rad.phase_b import b2_dlcm_official as official
    from rad.phase_b import b2_dlcm_target_conflict_diagnosis as diag
    from rad.phase_b import b2_dlcm_training as training
    from tests.rad.b2_dlcm_fixtures import ACCEPTED_UPSTREAM

    torch.use_deterministic_algorithms(True, warn_only=False)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    diag.guard_no_accepted_manifest_write(
        output_dir=output_dir, filename="diagnosis_report.md"
    )
    diag.guard_no_training_invocation(attempt_train=False)
    diag.guard_no_teacher_backbone_invocation(teacher_forward_count=0)

    # Freeze original checkpoint bytes.
    seed_ckpt_paths = {
        int(s): run_dir / f"seed_{s}" / "committed" / "best_training_checkpoint.pt"
        for s in config["seeds"]
    }
    last_ckpt_paths = {
        int(s): run_dir / f"seed_{s}" / "committed" / "last_training_checkpoint.pt"
        for s in config["seeds"]
    }
    deploy_path = run_dir / "canonical_deployment_checkpoint.pt"
    frozen_bytes = {str(p): p.read_bytes() for p in list(seed_ckpt_paths.values()) + list(last_ckpt_paths.values()) + [deploy_path]}
    frozen_hashes = {k: hashlib.sha256(v).hexdigest() for k, v in frozen_bytes.items()}

    # Qualification freeze from disk (immutable evidence).
    qual = json.loads((run_dir / "qualification_result.json").read_text(encoding="utf-8"))
    selection = json.loads((run_dir / "canonical_seed_selection.json").read_text(encoding="utf-8"))
    unlock = json.loads((run_dir / "evaluation_unlock.json").read_text(encoding="utf-8"))
    collection = json.loads((run_dir / "seed_collection_manifest.json").read_text(encoding="utf-8"))
    evidence_old = json.loads(
        (_REPO_ROOT / "docs/phase_b/b2_05b_dlcm_training_manifest.json").read_text(encoding="utf-8")
    )
    accepted_exists = (run_dir / "accepted_deployment_manifest.json").is_file()
    frozen_obs = {
        "qualification_status": qual["gates"]["state"],
        "deployment_qualified": bool(qual["gates"]["deployment_qualified"]),
        "accepted_deployment_manifest_created": accepted_exists,
        "identities": {
            "accepted_training_plan": evidence_old["accepted_dlcm_training_plan_scientific_sha256"],
            "seed_collection": collection["seed_collection_scientific_sha256"],
            "canonical_selection": selection["selection_scientific_sha256"],
            "deployment_scientific": qual["deployment_scientific_sha256"],
            "evaluation_unlock": unlock["evaluation_unlock_scientific_sha256"],
            "qualification_scientific": qual["qualification_scientific_sha256"],
        },
    }
    diag.assert_qualification_frozen(frozen_obs)

    if args.dry_run:
        print(json.dumps({"dry_run": True, "frozen": frozen_obs}, sort_keys=True, indent=2))
        return 0

    verified = training.load_verified_b2_dlcm_training_inputs(
        descriptor_manifest=args.descriptor_root / "final_manifest.json",
        descriptor_root=args.descriptor_root,
        contribution_target_manifest=args.contribution_target_root / "final_manifest.json",
        contribution_target_root=args.contribution_target_root,
        accepted_upstream=dict(config.get("accepted_upstream", ACCEPTED_UPSTREAM)),
        evaluation_unlocked=True,
    )
    plan_sha = official.compute_accepted_dlcm_training_plan_scientific_sha256(
        config=config, verified=verified
    )
    if plan_sha != diag.FROZEN_IDENTITIES["accepted_training_plan"]:
        raise SystemExit(f"plan sha drift: {plan_sha}")

    splits = {
        "training": list(verified.training_records),
        "calibration": list(verified.calibration_records),
        "evaluation": list(verified.require_evaluation_records()),
    }

    # Teacher labels (cached records only; no teacher forward).
    labels: dict[str, dict[str, Any]] = {}
    for split_name, rows in splits.items():
        for rec in rows:
            sid = str(rec["stable_sample_id"])
            payload = torch.load(
                args.teacher_cache_root / "samples" / f"{sid}.pt",
                map_location="cpu",
                weights_only=False,
            )
            sr = payload["scientific_record"]
            labels[sid] = {
                "image_label": int(sr["image_label"]),
                "anomaly_type": str(sr.get("anomaly_type", "")),
                "category": str(sr.get("category", rec["category"])),
                "split": split_name,
            }

    candidate_layers = tuple(config["candidate_layers"])
    depths = tuple(int(d) for d in config["prediction_depths"])

    # Load models (read-only).
    models: dict[int, Any] = {}
    for seed, path in seed_ckpt_paths.items():
        models[seed] = evaluation.load_training_model_from_best_checkpoint(path, seed=seed)
    canon_seed = int(args.seed)
    canon_model = models[canon_seed]
    # Verify deployment export identity source.
    deploy_ckpt = torch.load(deploy_path, map_location="cpu", weights_only=False)
    exported = dlcm.extract_deployment_state_dict(canon_model)
    deploy_sd = deploy_ckpt["state_dict"]
    export_match = exported.keys() == deploy_sd.keys() and all(
        torch.equal(exported[k].cpu(), deploy_sd[k].cpu()) for k in exported
    )

    from rad.phase_b import b2_dlcm_deployment as deployment

    # --- Per split/depth fidelity for trained canonical model + oracles ---
    fidelity_by_split: dict[str, Any] = {}
    conflict_by_split: dict[str, Any] = {}
    oracle_by_split: dict[str, Any] = {}
    trained_vs_oracle: dict[str, Any] = {}
    alpha_by_split: dict[str, Any] = {}

    for split_name, records in splits.items():
        fidelity_by_split[split_name] = {}
        conflict_by_split[split_name] = {}
        oracle_by_split[split_name] = {}
        trained_vs_oracle[split_name] = {}
        for depth in depths:
            trained_rows: list[dict[str, Any]] = []
            conflict_rows: list[dict[str, Any]] = []
            vs_rows: list[dict[str, Any]] = []
            for rec in records:
                p_gt = rec["p_gt"][depth].to(torch.float64)
                p_t = rec["p_t"][depth].to(torch.float64)
                phi_gt = rec["phi_gt"][depth].to(torch.float64)
                phi_t = rec["phi_t"][depth].to(torch.float64)
                w = diag.predict_model_weights_on_record(
                    canon_model, rec, depth=depth, candidate_layers=candidate_layers
                )
                eq = diag.equal_family_oracle(p_gt, p_t)
                metrics = diag.oracle_candidate_metrics(p_gt=p_gt, p_t=p_t, weights=w)
                sid = str(rec["stable_sample_id"])
                meta = labels[sid]
                trained_rows.append(
                    {
                        "stable_sample_id": sid,
                        "category": str(rec["category"]),
                        "image_label": meta["image_label"],
                        "anomaly_type": meta["anomaly_type"],
                        "allocation_kl_gt": metrics["kl_gt"],
                        "allocation_kl_teacher": metrics["kl_teacher"],
                        "allocation_kl_gt_uniform": metrics["kl_gt_uniform"],
                        "allocation_kl_teacher_uniform": metrics["kl_teacher_uniform"],
                        "delta_from_uniform_gt": metrics["delta_from_uniform_gt"],
                        "delta_from_uniform_teacher": metrics["delta_from_uniform_teacher"],
                        "allocation_jsd_gt": float(
                            deployment.allocation_jsd(p_gt.float(), w.float()).item()
                        ),
                        "allocation_jsd_teacher": float(
                            deployment.allocation_jsd(p_t.float(), w.float()).item()
                        ),
                        "allocation_top1_gt": deployment.top1_set_agreement(p_gt.float(), w.float()),
                        "allocation_top1_teacher": deployment.top1_set_agreement(
                            p_t.float(), w.float()
                        ),
                        "allocation_spearman_gt": deployment.spearman_average_ranks(
                            w.float(), p_gt.float()
                        ),
                        "allocation_spearman_teacher": deployment.spearman_average_ranks(
                            w.float(), p_t.float()
                        ),
                    }
                )
                conflict_rows.append(
                    {
                        "stable_sample_id": sid,
                        "category": str(rec["category"]),
                        "image_label": meta["image_label"],
                        **diag.target_conflict_stats(p_gt, p_t, phi_gt=phi_gt, phi_t=phi_t),
                    }
                )
                vs_rows.append(
                    {
                        "stable_sample_id": sid,
                        "category": str(rec["category"]),
                        "image_label": meta["image_label"],
                        **diag.compare_weights_to_targets(
                            w, p_gt=p_gt, p_t=p_t, equal_oracle=eq
                        ),
                    }
                )

            fidelity_by_split[split_name][depth] = {
                "gt": diag.summarize_rows_by_category(
                    [
                        {
                            "stable_sample_id": r["stable_sample_id"],
                            "category": r["category"],
                            "allocation_kl": r["allocation_kl_gt"],
                            "allocation_kl_uniform": r["allocation_kl_gt_uniform"],
                            "delta_from_uniform": r["delta_from_uniform_gt"],
                            "allocation_jsd": r["allocation_jsd_gt"],
                            "allocation_top1": r["allocation_top1_gt"],
                            "allocation_spearman": r["allocation_spearman_gt"],
                        }
                        for r in trained_rows
                    ],
                    keys=(
                        "allocation_kl",
                        "allocation_kl_uniform",
                        "delta_from_uniform",
                        "allocation_jsd",
                        "allocation_top1",
                        "allocation_spearman",
                    ),
                ),
                "teacher": diag.summarize_rows_by_category(
                    [
                        {
                            "stable_sample_id": r["stable_sample_id"],
                            "category": r["category"],
                            "allocation_kl": r["allocation_kl_teacher"],
                            "allocation_kl_uniform": r["allocation_kl_teacher_uniform"],
                            "delta_from_uniform": r["delta_from_uniform_teacher"],
                            "allocation_jsd": r["allocation_jsd_teacher"],
                            "allocation_top1": r["allocation_top1_teacher"],
                            "allocation_spearman": r["allocation_spearman_teacher"],
                        }
                        for r in trained_rows
                    ],
                    keys=(
                        "allocation_kl",
                        "allocation_kl_uniform",
                        "delta_from_uniform",
                        "allocation_jsd",
                        "allocation_top1",
                        "allocation_spearman",
                    ),
                ),
                "equal_family_diagnostic_average": diag.summarize_rows_by_category(
                    [
                        {
                            "stable_sample_id": r["stable_sample_id"],
                            "category": r["category"],
                            "dual_kl": 0.5 * r["allocation_kl_gt"]
                            + 0.5 * r["allocation_kl_teacher"],
                        }
                        for r in trained_rows
                    ],
                    keys=("dual_kl",),
                ),
                "per_sample": trained_rows,
            }

            conflict_by_split[split_name][depth] = diag.summarize_rows_by_category(
                conflict_rows,
                keys=(
                    "jsd_natural_log",
                    "l1",
                    "l2",
                    "target_top1_agreement",
                    "target_spearman",
                    "entropy_gt",
                    "entropy_teacher",
                    "max_prob_gt",
                    "max_prob_teacher",
                    "support_overlap",
                    "signed_top1_agreement",
                    "signed_spearman",
                ),
            )
            conflict_by_split[split_name][depth]["fraction_teacher_flatter"] = sum(
                1 for r in conflict_rows if r["teacher_flatter_than_gt"]
            ) / max(1, len(conflict_rows))
            conflict_by_split[split_name][depth]["fraction_teacher_more_uniform"] = sum(
                1 for r in conflict_rows if r["teacher_more_uniform_than_gt"]
            ) / max(1, len(conflict_rows))

            oracle_by_split[split_name][depth] = {
                "uniform": diag.evaluate_weight_candidates_on_records(
                    records,
                    depth=depth,
                    weight_fn=lambda p_gt, p_t: diag.uniform_weights(int(p_gt.numel())),
                ),
                "gt_oracle": diag.evaluate_weight_candidates_on_records(
                    records, depth=depth, weight_fn=lambda p_gt, p_t: p_gt
                ),
                "teacher_oracle": diag.evaluate_weight_candidates_on_records(
                    records, depth=depth, weight_fn=lambda p_gt, p_t: p_t
                ),
                "equal_family_oracle": diag.evaluate_weight_candidates_on_records(
                    records, depth=depth, weight_fn=diag.equal_family_oracle
                ),
            }
            tm_summary = diag.summarize_rows_by_category(
                [
                    {
                        "stable_sample_id": r["stable_sample_id"],
                        "category": r["category"],
                        "kl_gt": r["allocation_kl_gt"],
                        "kl_teacher": r["allocation_kl_teacher"],
                        "kl_gt_uniform": r["allocation_kl_gt_uniform"],
                        "kl_teacher_uniform": r["allocation_kl_teacher_uniform"],
                        "dual_family_mean_kl": 0.5 * r["allocation_kl_gt"]
                        + 0.5 * r["allocation_kl_teacher"],
                        "delta_from_uniform_gt": r["delta_from_uniform_gt"],
                        "delta_from_uniform_teacher": r["delta_from_uniform_teacher"],
                        "gt_improvement_over_uniform": -r["delta_from_uniform_gt"],
                        "teacher_improvement_over_uniform": -r["delta_from_uniform_teacher"],
                    }
                    for r in trained_rows
                ],
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
            tm_summary["both_target_learning_gates_pass"] = diag.family_gates_pass(
                kl_gt_macro=float(tm_summary["category_macro"]["kl_gt"]),
                kl_gt_uniform_macro=float(tm_summary["category_macro"]["kl_gt_uniform"]),
                kl_teacher_macro=float(tm_summary["category_macro"]["kl_teacher"]),
                kl_teacher_uniform_macro=float(tm_summary["category_macro"]["kl_teacher_uniform"]),
                per_category={
                    cat: {
                        "gt": float(tm_summary["per_category"]["kl_gt"][cat]),
                        "gt_uniform": float(tm_summary["per_category"]["kl_gt_uniform"][cat]),
                        "teacher": float(tm_summary["per_category"]["kl_teacher"][cat]),
                        "teacher_uniform": float(
                            tm_summary["per_category"]["kl_teacher_uniform"][cat]
                        ),
                    }
                    for cat in tm_summary["per_category"]["kl_gt"]
                },
            )
            oracle_by_split[split_name][depth]["trained_model"] = tm_summary
            trained_vs_oracle[split_name][depth] = diag.summarize_rows_by_category(
                vs_rows,
                keys=(
                    "jsd_vs_gt",
                    "jsd_vs_teacher",
                    "jsd_vs_equal_oracle",
                    "jsd_vs_uniform",
                    "l1_vs_gt",
                    "l1_vs_teacher",
                    "l1_vs_equal_oracle",
                    "allocation_top1_vs_gt",
                    "allocation_top1_vs_teacher",
                    "allocation_top1_vs_equal_oracle",
                    "spearman_vs_gt",
                    "spearman_vs_teacher",
                    "spearman_vs_equal_oracle",
                    "entropy_w",
                    "max_weight",
                ),
            )

        alpha_raw = diag.alpha_feasibility_on_records(records, depth=24)
        alpha_by_split[split_name] = {
            "feasible_alphas": alpha_raw["feasible_alphas"],
            "interval": alpha_raw["interval"],
            "n_feasible": len(alpha_raw["feasible_alphas"]),
            "per_alpha_compact": [
                a
                for a in alpha_raw["per_alpha"]
                if a["alpha"] in {0.0, 0.25, 0.5, 0.75, 1.0} or a["both_gates_pass"]
            ],
            "full_per_alpha": alpha_raw["per_alpha"],
        }

    alpha_report = diag.finalize_alpha_feasibility_report(
        calibration={
            "feasible_alphas": alpha_by_split["calibration"]["feasible_alphas"],
            "interval": alpha_by_split["calibration"]["interval"],
        },
        evaluation_posthoc={
            "feasible_alphas": alpha_by_split["evaluation"]["feasible_alphas"],
            "interval": alpha_by_split["evaluation"]["interval"],
        },
    )

    # --- Seed-by-seed evaluation GT/teacher (diagnostic only) ---
    seed_eval: dict[str, Any] = {}
    for seed, model in models.items():
        # Prefer sealed evaluation JSON if present (immutable), recompute only for cross-check.
        sealed = run_dir / "evaluation" / f"seed_{seed}_evaluation.json"
        sealed_payload = json.loads(sealed.read_text(encoding="utf-8"))
        d24 = sealed_payload["depth_results"]["24"]["target_fidelity"]
        seed_eval[str(seed)] = {
            "source": "sealed_evaluation_json",
            "kl_gt_macro": d24["kl_gt_dlcm"]["category_macro"],
            "kl_gt_uniform_macro": d24["kl_gt_uniform"]["category_macro"],
            "kl_teacher_macro": d24["kl_teacher_dlcm"]["category_macro"],
            "kl_teacher_uniform_macro": d24["kl_teacher_uniform"]["category_macro"],
            "delta_gt": d24["kl_gt_dlcm"]["category_macro"] - d24["kl_gt_uniform"]["category_macro"],
            "delta_teacher": d24["kl_teacher_dlcm"]["category_macro"]
            - d24["kl_teacher_uniform"]["category_macro"],
            "per_category_gt": d24["kl_gt_dlcm"]["per_category"],
            "per_category_teacher": d24["kl_teacher_dlcm"]["per_category"],
            "teacher_fails_macro": d24["kl_teacher_dlcm"]["category_macro"]
            > d24["kl_teacher_uniform"]["category_macro"] - 1e-5,
        }

    # Calibration ranking vs teacher evaluation
    seed_manifests = {}
    for seed in config["seeds"]:
        seed_manifests[str(seed)] = json.loads(
            (run_dir / f"seed_{seed}" / "seed_manifest.json").read_text(encoding="utf-8")
        )
    ranking = sorted(
        config["seeds"],
        key=lambda s: float(seed_manifests[str(s)]["calibration_primary"]),
    )
    teacher_eval_order = sorted(
        config["seeds"],
        key=lambda s: float(seed_eval[str(s)]["kl_teacher_macro"]),
    )

    # Trace epochs for canonical seed
    trace_path = run_dir / f"seed_{canon_seed}" / "committed" / "training_trace.json"
    best_epoch = int(seed_manifests[str(canon_seed)]["best_epoch"])
    last_epoch = max(
        int(n["record"]["epoch"])
        for n in json.loads(trace_path.read_text(encoding="utf-8"))["nodes"]
    )
    trace_epochs = _load_trace_epochs(trace_path, [0, best_epoch, last_epoch])

    # Selector diagnostic on calibration with each seed best checkpoint
    selector_diag: dict[str, Any] = {}
    cal_records = splits["calibration"]
    seed_cal_scores: dict[str, Any] = {}
    for seed, model in models.items():
        rows = []
        for rec in cal_records:
            p_gt = rec["p_gt"][24].to(torch.float64)
            p_t = rec["p_t"][24].to(torch.float64)
            w = diag.predict_model_weights_on_record(
                model, rec, depth=24, candidate_layers=candidate_layers
            )
            m = diag.oracle_candidate_metrics(p_gt=p_gt, p_t=p_t, weights=w)
            rows.append({"category": rec["category"], **m})
        # category macro
        cats = sorted({r["category"] for r in rows})
        kl_gt = sum(
            sum(r["kl_gt"] for r in rows if r["category"] == c) / max(1, sum(1 for r in rows if r["category"] == c))
            for c in cats
        ) / max(1, len(cats))
        kl_t = sum(
            sum(r["kl_teacher"] for r in rows if r["category"] == c)
            / max(1, sum(1 for r in rows if r["category"] == c))
            for c in cats
        ) / max(1, len(cats))
        kl_gt_u = sum(
            sum(r["kl_gt_uniform"] for r in rows if r["category"] == c)
            / max(1, sum(1 for r in rows if r["category"] == c))
            for c in cats
        ) / max(1, len(cats))
        kl_t_u = sum(
            sum(r["kl_teacher_uniform"] for r in rows if r["category"] == c)
            / max(1, sum(1 for r in rows if r["category"] == c))
            for c in cats
        ) / max(1, len(cats))
        scores = diag.diagnostic_selector_scores(
            kl_gt=kl_gt,
            kl_teacher=kl_t,
            kl_gt_uniform=kl_gt_u,
            kl_teacher_uniform=kl_t_u,
        )
        seed_cal_scores[str(seed)] = {
            "kl_gt_macro": kl_gt,
            "kl_teacher_macro": kl_t,
            "kl_gt_uniform_macro": kl_gt_u,
            "kl_teacher_uniform_macro": kl_t_u,
            "calibration_primary_manifest": float(seed_manifests[str(seed)]["calibration_primary"]),
            "calibration_secondary_manifest": float(seed_manifests[str(seed)]["calibration_secondary"]),
            **scores,
        }

    def _pick(metric_key: str, *, maximize: bool = False) -> int:
        items = [(int(s), seed_cal_scores[s][metric_key]) for s in seed_cal_scores]
        items.sort(key=lambda x: (x[1] if not maximize else -x[1], x[0]))
        return items[0][0]

    selector_diag = {
        "frozen_selector_canonical_seed": canon_seed,
        "mean_family_kl_would_select": _pick("mean_family_kl"),
        "worst_family_kl_would_select": _pick("worst_family_kl"),
        "uniform_relative_worst_family_delta_would_select": _pick(
            "uniform_relative_worst_family_delta"
        ),
        "constrained_selector_would_select": (
            _pick("constrained_objective")
            if any(seed_cal_scores[s]["constrained_feasible"] for s in seed_cal_scores)
            else None
        ),
        "per_seed_calibration_scores": seed_cal_scores,
        "note": "diagnostic only; canonical seed unchanged",
    }

    # Normal/anomalous breakdown on evaluation depth 24
    eval_rows = fidelity_by_split["evaluation"][24]["per_sample"]
    by_label: dict[str, Any] = {"normal": [], "anomalous": []}
    for r in eval_rows:
        bucket = "anomalous" if int(r["image_label"]) == 1 else "normal"
        by_label[bucket].append(r)
    normal_anom = {
        label: {
            "n": len(rows),
            "mean_kl_gt": sum(r["allocation_kl_gt"] for r in rows) / max(1, len(rows)),
            "mean_kl_teacher": sum(r["allocation_kl_teacher"] for r in rows) / max(1, len(rows)),
            "mean_delta_gt": sum(r["delta_from_uniform_gt"] for r in rows) / max(1, len(rows)),
            "mean_delta_teacher": sum(r["delta_from_uniform_teacher"] for r in rows)
            / max(1, len(rows)),
        }
        for label, rows in by_label.items()
    }

    # Classification (A–E). Tags may accumulate → primary E when mixed.
    cal_eq = oracle_by_split["calibration"][24]["equal_family_oracle"]
    eval_eq = oracle_by_split["evaluation"][24]["equal_family_oracle"]
    cal_trained = oracle_by_split["calibration"][24]["trained_model"]
    eval_trained = oracle_by_split["evaluation"][24]["trained_model"]
    cal_alpha_feasible = bool(alpha_by_split["calibration"]["feasible_alphas"])
    eval_alpha_feasible = bool(alpha_by_split["evaluation"]["feasible_alphas"])

    classifications: list[str] = []
    evidence_points: list[str] = []

    # A: frozen equal-family (α=0.5) single-head objective infeasible on calibration.
    if not cal_eq["both_target_learning_gates_pass"]:
        classifications.append("A")
        evidence_points.append(
            "Equal-family oracle fails both target-learning gates on calibration "
            f"(GT macro KL={cal_eq['category_macro']['kl_gt']:.4f}, "
            f"teacher macro KL={cal_eq['category_macro']['kl_teacher']:.4f}; "
            f"uniform teacher={cal_eq['category_macro']['kl_teacher_uniform']:.4f})."
        )
        if not cal_alpha_feasible:
            evidence_points.append(
                "No calibration-feasible alpha exists on the declared grid either "
                "(Case A for any fixed family weighting)."
            )
        else:
            evidence_points.append(
                "However a non-equal calibration-feasible alpha interval exists "
                f"{alpha_by_split['calibration']['interval']} (Case B for reweighted "
                "single-head family balance; must not be chosen from evaluation)."
            )

    # B: equal-family feasible on calibration, but training/selection failed.
    if cal_eq["both_target_learning_gates_pass"] and not cal_trained[
        "both_target_learning_gates_pass"
    ]:
        classifications.append("B")
        evidence_points.append(
            "Equal-family oracle passes calibration gates but the trained model does not."
        )

    # C: feasible on calibration path, evaluation generalization fails.
    if cal_alpha_feasible and not eval_alpha_feasible:
        classifications.append("C")
        evidence_points.append(
            "Calibration-feasible alpha interval exists, but evaluation post-hoc "
            "feasible alpha interval is empty (generalization / small-sample conflict)."
        )
    elif (
        cal_eq["both_target_learning_gates_pass"]
        and cal_trained.get("both_target_learning_gates_pass", False)
        and not eval_trained["both_target_learning_gates_pass"]
    ):
        classifications.append("C")
        evidence_points.append(
            "Trained model passes calibration target gates but fails evaluation."
        )
    elif cal_eq["both_target_learning_gates_pass"] and not eval_eq[
        "both_target_learning_gates_pass"
    ]:
        classifications.append("C")
        evidence_points.append(
            "Equal-family oracle passes calibration but fails evaluation."
        )

    # D: category / sample-type concentration.
    eval_gt_pc = fidelity_by_split["evaluation"][24]["gt"]["per_category"]["allocation_kl"]
    eval_t_pc = fidelity_by_split["evaluation"][24]["teacher"]["per_category"]["allocation_kl"]
    eval_t_u = fidelity_by_split["evaluation"][24]["teacher"]["per_category"][
        "allocation_kl_uniform"
    ]
    failing_cats = [c for c in eval_t_pc if eval_t_pc[c] > eval_t_u[c] + 1e-4]
    if failing_cats and len(failing_cats) < len(eval_t_pc):
        classifications.append("D")
        evidence_points.append(f"Teacher failure concentrated in categories: {failing_cats}")
    elif failing_cats:
        evidence_points.append(f"Teacher fails in all eval categories: {failing_cats}")

    # All seeds fail teacher on sealed evaluation.
    if all(seed_eval[str(s)]["teacher_fails_macro"] for s in config["seeds"]):
        evidence_points.append(
            "All seeds {17,29,43} fail teacher macro KL vs uniform on sealed evaluation."
        )

    classifications = list(dict.fromkeys(classifications))  # stable unique
    if len(classifications) > 1:
        primary = "E"
        evidence_points.append(f"Mixed causes flagged: {classifications}")
    elif len(classifications) == 1:
        primary = classifications[0]
    else:
        primary = "E"
        evidence_points.append("No exclusive A–D match; defaulting to mixed.")

    # Signed diagnostics from training checkpoint (not deployment)
    signed_diag = evaluation.compute_signed_diagnostics_report(
        model=canon_model,
        descriptors=splits["evaluation"][0]["descriptors"][24],
        prediction_depth=24,
        player_layer_ids=dlcm.players_for_depth(candidate_layers, 24),
        phi_gt=splits["evaluation"][0]["phi_gt"][24],
        phi_t=splits["evaluation"][0]["phi_t"][24],
        artifact_kind="training",
        diagnostic_source="canonical_best_training_checkpoint",
    )
    deploy_model = evaluation.load_deployment_trunk_from_checkpoint(deploy_path)
    signed_deploy = evaluation.compute_signed_diagnostics_report(
        model=deploy_model,
        descriptors=splits["evaluation"][0]["descriptors"][24],
        prediction_depth=24,
        player_layer_ids=dlcm.players_for_depth(candidate_layers, 24),
        phi_gt=splits["evaluation"][0]["phi_gt"][24],
        phi_t=splits["evaluation"][0]["phi_t"][24],
        artifact_kind="deployment",
    )

    # Re-verify checkpoint immutability
    for path_str, expected in frozen_bytes.items():
        diag.assert_checkpoint_bytes_immutable(path_str, expected_bytes=expected)

    # Compact alpha for JSON (drop full_per_alpha from split copies into top-level only)
    alpha_compact = {
        split: {
            "feasible_alphas": alpha_by_split[split]["feasible_alphas"],
            "interval": alpha_by_split[split]["interval"],
            "n_feasible": alpha_by_split[split]["n_feasible"],
            "per_alpha_compact": alpha_by_split[split]["per_alpha_compact"],
        }
        for split in alpha_by_split
    }

    # Strip bulky per_sample from top-level optional? Keep but maybe trim in md.
    payload: dict[str, Any] = {
        "schema_version": "b2_05c0_target_conflict_diagnosis_v1",
        "diagnosis_only": True,
        "no_training_performed": True,
        "no_checkpoint_modified": True,
        "no_accepted_manifest_generated": True,
        "no_teacher_backbone_rerun": True,
        "canonical_seed": canon_seed,
        "authoritative_run": run_dir.name,
        "deployment_export_matches_best_training": export_match,
        "signed_diagnostics_boundary": {
            "training_checkpoint_sample": {
                "diagnostic_source": signed_diag.get("diagnostic_source"),
                "not_part_of_deployment_artifact": signed_diag.get("not_part_of_deployment_artifact"),
                "signed_diagnostics_available": signed_diag.get("signed_diagnostics_available"),
                "huber_gt_status": signed_diag["huber_gt"]["status"],
            },
            "deployment_checkpoint_sample": {
                "signed_diagnostics_available": signed_deploy.get("signed_diagnostics_available"),
                "huber_gt": signed_deploy["huber_gt"],
            },
            "correction": "deployment weights are never used as signed Shapley proxies",
        },
        "frozen_qualification": frozen_obs,
        "checkpoint_sha256_before_after_unchanged": frozen_hashes,
        "fidelity_by_split_depth": {
            split: {
                str(depth): {
                    "gt_category_macro": fidelity_by_split[split][depth]["gt"]["category_macro"],
                    "teacher_category_macro": fidelity_by_split[split][depth]["teacher"][
                        "category_macro"
                    ],
                    "gt_per_category": fidelity_by_split[split][depth]["gt"]["per_category"],
                    "teacher_per_category": fidelity_by_split[split][depth]["teacher"][
                        "per_category"
                    ],
                    "equal_family_diagnostic_average_macro": fidelity_by_split[split][depth][
                        "equal_family_diagnostic_average"
                    ]["category_macro"],
                }
                for depth in depths
            }
            for split in fidelity_by_split
        },
        "fidelity_per_sample_evaluation_depth24": fidelity_by_split["evaluation"][24]["per_sample"],
        "target_conflict_by_split_depth": {
            split: {
                str(depth): {
                    "category_macro": conflict_by_split[split][depth]["category_macro"],
                    "per_category": conflict_by_split[split][depth]["per_category"],
                    "fraction_teacher_flatter": conflict_by_split[split][depth][
                        "fraction_teacher_flatter"
                    ],
                    "fraction_teacher_more_uniform": conflict_by_split[split][depth][
                        "fraction_teacher_more_uniform"
                    ],
                }
                for depth in depths
            }
            for split in conflict_by_split
        },
        "oracles_by_split_depth24": {
            split: {
                name: {
                    "category_macro": oracle_by_split[split][24][name]["category_macro"],
                    "per_category": {
                        k: oracle_by_split[split][24][name]["per_category"][k]
                        for k in (
                            "kl_gt",
                            "kl_teacher",
                            "delta_from_uniform_gt",
                            "delta_from_uniform_teacher",
                        )
                        if k in oracle_by_split[split][24][name]["per_category"]
                    },
                    "both_target_learning_gates_pass": oracle_by_split[split][24][name].get(
                        "both_target_learning_gates_pass"
                    ),
                    "per_category_max_degradation_vs_uniform": oracle_by_split[split][24][name].get(
                        "per_category_max_degradation_vs_uniform"
                    ),
                }
                for name in (
                    "uniform",
                    "gt_oracle",
                    "teacher_oracle",
                    "equal_family_oracle",
                    "trained_model",
                )
            }
            for split in oracle_by_split
        },
        "trained_vs_equal_oracle_by_split_depth": {
            split: {
                str(depth): trained_vs_oracle[split][depth]["category_macro"]
                for depth in depths
            }
            for split in trained_vs_oracle
        },
        "alpha_feasibility": alpha_report,
        "alpha_by_split_depth24": alpha_compact,
        "seed_evaluation_depth24": seed_eval,
        "seed_calibration_ranking_order": ranking,
        "seed_teacher_eval_order_ascending_kl": teacher_eval_order,
        "calibration_primary_correlates_with_better_teacher_eval": ranking
        == teacher_eval_order,
        "canonical_trace_epochs": trace_epochs,
        "selector_diagnostic": selector_diag,
        "normal_anomalous_evaluation_depth24": normal_anom,
        "category_concentration_evaluation_depth24": {
            "gt_kl": eval_gt_pc,
            "teacher_kl": eval_t_pc,
            "teacher_uniform_kl": eval_t_u,
            "teacher_failing_categories": failing_cats,
        },
        "classification": {
            "primary": primary,
            "supporting_tags": classifications,
            "evidence": evidence_points,
            "single_head_fixed_family_weighting_appears_feasible_on_calibration": cal_alpha_feasible,
            "equal_family_objective_feasible_on_calibration": cal_eq[
                "both_target_learning_gates_pass"
            ],
            "alpha_case": (
                "A_no_calibration_feasible_alpha"
                if not cal_alpha_feasible
                else (
                    "C_calibration_feasible_evaluation_infeasible"
                    if not eval_alpha_feasible
                    else "B_calibration_feasible_alpha_exists"
                )
            ),
        },
        "evaluation_contamination_boundary": {
            "current_evaluation_split_status": "used_for_B2_05B_qualification_and_postmortem",
            "protocols": {
                "protocol_1": "current evaluation becomes development; create new untouched final evaluation",
                "protocol_2": "no model change; preserve unqualified result; do not start LSE",
            },
            "forbidden": "retrain and requalify on the same eight evaluation records as untouched test",
        },
    }

    # Drop full_per_alpha from disk payload size: already compact; keep full under alpha_by_split files? 
    # Store full alpha grid separately compactly at top level for calibration/evaluation
    payload["alpha_grid_full_calibration"] = alpha_by_split["calibration"]["full_per_alpha"]
    payload["alpha_grid_full_evaluation"] = alpha_by_split["evaluation"]["full_per_alpha"]
    payload["scientific_content_sha256"] = _canonical_json_sha256(
        {k: v for k, v in payload.items() if k != "scientific_content_sha256"}
    )

    json_path = output_dir / "b2_05c0_target_conflict_diagnosis.json"
    md_path = output_dir / "b2_05c0_target_conflict_diagnosis.md"
    json_sha = _write_json(json_path, payload)

    # Markdown report
    d24_eval_gt = fidelity_by_split["evaluation"][24]["gt"]["category_macro"]
    d24_eval_t = fidelity_by_split["evaluation"][24]["teacher"]["category_macro"]
    eq_cal = oracle_by_split["calibration"][24]["equal_family_oracle"]["category_macro"]
    eq_eval = oracle_by_split["evaluation"][24]["equal_family_oracle"]["category_macro"]
    md = f"""# B2-05C0 Target-Conflict Diagnosis

## Status

- Diagnosis only: **no retraining**, no checkpoint mutation, no accepted manifest
- Qualification unchanged: `{frozen_obs['qualification_status']}`
- Deployment qualified: `{frozen_obs['deployment_qualified']}`
- Classification: **{primary}**
- Single-head fixed-family weighting feasible on calibration: **{cal_alpha_feasible}**

## Signed-diagnostics boundary correction

- Deployment weights are **not** valid signed-Shapley proxies
- Deployment signed metrics: `not_available_in_deployment_artifact`
- Canonical signed diagnostics bind to `canonical_best_training_checkpoint` with `not_part_of_deployment_artifact=true`
- Deployment export matches best training trunk: `{export_match}`

## Frozen identities (unchanged)

| Identity | SHA-256 |
|----------|---------|
| accepted training plan | `{frozen_obs['identities']['accepted_training_plan']}` |
| seed collection | `{frozen_obs['identities']['seed_collection']}` |
| canonical selection | `{frozen_obs['identities']['canonical_selection']}` |
| deployment scientific | `{frozen_obs['identities']['deployment_scientific']}` |
| evaluation unlock | `{frozen_obs['identities']['evaluation_unlock']}` |
| qualification scientific | `{frozen_obs['identities']['qualification_scientific']}` |

## Evaluation depth-24 trained model (canonical seed {canon_seed})

| Family | DLCM KL macro | Uniform KL macro | delta (DLCM−uniform) |
|--------|---------------|------------------|----------------------|
| GT | {d24_eval_gt['allocation_kl']:.6f} | {d24_eval_gt['allocation_kl_uniform']:.6f} | {d24_eval_gt['delta_from_uniform']:.6f} |
| Teacher | {d24_eval_t['allocation_kl']:.6f} | {d24_eval_t['allocation_kl_uniform']:.6f} | {d24_eval_t['delta_from_uniform']:.6f} |

Per-category teacher KL (eval d24): `{eval_t_pc}`
Per-category GT KL (eval d24): `{eval_gt_pc}`

## Equal-family oracle (most important)

| Split | GT KL macro | Teacher KL macro | both gates pass |
|-------|-------------|------------------|-----------------|
| calibration | {eq_cal['kl_gt']:.6f} | {eq_cal['kl_teacher']:.6f} | {oracle_by_split['calibration'][24]['equal_family_oracle']['both_target_learning_gates_pass']} |
| evaluation | {eq_eval['kl_gt']:.6f} | {eq_eval['kl_teacher']:.6f} | {oracle_by_split['evaluation'][24]['equal_family_oracle']['both_target_learning_gates_pass']} |

## Alpha feasibility

- Calibration feasible alpha interval: `{alpha_report['calibration_feasible_alpha_interval']}`
- Evaluation post-hoc feasible alpha interval (diagnostic only): `{alpha_report['evaluation_posthoc_feasible_alpha_interval']}`
- Evaluation **not** used to choose alpha

## Seed-by-seed evaluation (sealed JSON)

{json.dumps(seed_eval, indent=2, sort_keys=True)}

## Selector diagnostic (calibration, read-only)

{json.dumps(selector_diag, indent=2, sort_keys=True)}

## Normal / anomalous (eval d24)

{json.dumps(normal_anom, indent=2, sort_keys=True)}

## Classification evidence

{chr(10).join('- ' + e for e in evidence_points)}

## Evaluation contamination boundary

```
current evaluation split status =
used_for_B2_05B_qualification_and_postmortem
```

Any future change to family weights, selection, architecture, loss, targets, or thresholds must follow Protocol 1 (new untouched eval) or Protocol 2 (freeze unqualified; no LSE).

## Artifact

- JSON: `{json_path.name}` sha256 `{json_sha}`
"""
    md_path.write_text(md, encoding="utf-8")
    print(json.dumps({"json": str(json_path), "sha256": json_sha, "classification": primary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

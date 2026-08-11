#!/usr/bin/env python3
"""Authoritative B2-05C3B three-seed V3 uniform-relative DLCM training + development gates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_trace(seed_dir: Path) -> dict[str, Any]:
    return json.loads((seed_dir / "committed" / "training_trace.json").read_text(encoding="utf-8"))


def _teacher_alloc_diagnostics(model: Any, records: list[Any]) -> dict[str, Any]:
    import torch

    from rad.phase_b import b2_dlcm as v1
    from rad.phase_b import b2_dlcm_deployment as deployment

    model.eval()
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for record in records:
            for depth in model.prediction_depths:
                n = len(model.players_for_depth(int(depth)))
                desc = record.descriptors[int(depth)].unsqueeze(0)
                out = model.forward_training(desc, prediction_depth=int(depth))
                p_t = record.p_t[int(depth)].unsqueeze(0)
                kl = float(v1.allocation_kl(p_t, out.teacher_allocation_logits))
                w = out.teacher_allocation_weights.reshape(-1)
                pt = p_t.reshape(-1)
                rows.append(
                    {
                        "kl": kl,
                        "jsd": float(deployment.allocation_jsd(pt, w).item()),
                        "top1": float(deployment.top1_set_agreement(pt, w)),
                        "spearman": float(deployment.spearman_average_ranks(w, pt)),
                    }
                )
    return {
        "teacher_alloc_kl_macro": sum(r["kl"] for r in rows) / len(rows),
        "teacher_alloc_jsd_macro": sum(r["jsd"] for r in rows) / len(rows),
        "teacher_alloc_top1_macro": sum(r["top1"] for r in rows) / len(rows),
        "teacher_alloc_spearman_macro": sum(r["spearman"] for r in rows) / len(rows),
        "n": len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--descriptor-manifest", required=True)
    parser.add_argument("--descriptor-root", required=True)
    parser.add_argument("--contribution-target-manifest", required=True)
    parser.add_argument("--contribution-target-root", required=True)
    parser.add_argument("--teacher-cache-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--maximum-epochs", type=int, default=None)
    parser.add_argument("--skip-training", action="store_true", help="Resume from sealed seeds")
    args = parser.parse_args()

    import torch

    from rad.phase_b import b2_descriptor_artifacts as desc_mod
    from rad.phase_b import b2_dlcm_deployment as v1_deployment
    from rad.phase_b import b2_dlcm_evaluation as v1_evaluation
    from rad.phase_b import b2_dlcm_training as training
    from rad.phase_b import b2_dlcm_v4 as v4
    from rad.phase_b import b2_dlcm_v4_deployment as v4_deployment
    from rad.phase_b import b2_dlcm_v4_evaluation as v4_evaluation
    from rad.phase_b import b2_dlcm_v4_official as official
    from rad.phase_b import b2_dlcm_v4_protocol as protocol
    from rad.phase_b import b2_dlcm_v4_training as v4_training

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    official.verify_repository_identity_gate(config=config, repo_root=_REPO_ROOT)
    roster_payload, adoption_payload = official.load_frozen_roster_and_adoption(_REPO_ROOT)

    verified = training.load_verified_b2_dlcm_training_inputs(
        descriptor_manifest=args.descriptor_manifest,
        descriptor_root=args.descriptor_root,
        contribution_target_manifest=args.contribution_target_manifest,
        contribution_target_root=args.contribution_target_root,
        accepted_upstream=dict(config["accepted_upstream"]),
        evaluation_unlocked=False,
    )
    plan_sha = official.compute_accepted_v4_training_plan_scientific_sha256(
        config=config, verified=verified, roster_payload=roster_payload, adoption_payload=adoption_payload
    )
    official.require_plan_sha_agreement(
        config=config, recomputed=plan_sha, cli_expected=args.expected_plan_sha256
    )

    output_root = Path(args.output_root)
    if not args.skip_training:
        if output_root.exists():
            print(f"B2_DLCM_OUTPUT_COLLISION: {output_root} exists", file=sys.stderr)
            return 2
        output_root.mkdir(parents=True, exist_ok=False)

    env = training.collect_environment_contract(allow_cpu_for_hermetic=False)
    env_sha = training.persist_environment_contract(
        output_root / "environment_contract.json", env
    )
    train_ns = official.records_as_namespaces(verified.training_records)
    cal_ns = official.records_as_namespaces(verified.calibration_records)
    max_epochs = (
        int(args.maximum_epochs)
        if args.maximum_epochs is not None
        else int(config["maximum_epochs"])
    )

    seed_rows: list[dict[str, Any]] = []
    if not args.skip_training:
        for seed in list(config["seeds"]):
            print(f"=== starting seed {seed} ===", flush=True)
            result = official.run_official_v4_seed_training(
                output_root=output_root,
                seed=int(seed),
                training_records=train_ns,
                calibration_records=cal_ns,
                environment_contract=env,
                maximum_epochs=max_epochs,
                patience=int(config["patience"]),
                batch_size=int(config["batch_size"]),
                smoothmax_tau=float(config["smoothmax_tau"]),
                device="cuda",
            )
            if result["status"] == "failed":
                fail = training.build_collection_failure_manifest(
                    failed_seed=int(seed),
                    completed_seeds=[r["seed"] for r in seed_rows],
                    environment_identity=env_sha,
                )
                training.persist_collection_failure_manifest(
                    output_root / "collection_failure_manifest.json", fail
                )
                return 3
            seed_dir = output_root / f"seed_{seed}"
            best_pt = seed_dir / "committed" / "best_training_checkpoint.pt"
            seed_scientific = training._canonical_json_sha256(
                {
                    "seed": int(seed),
                    "best_epoch": result["best_epoch"],
                    "worst_relative_regret": float(result["worst_relative_regret"]),
                    "macro_kl": float(result["macro_kl"]),
                    "gt_signed": float(result["gt_signed"]),
                    "eligible": bool(result["eligible"]),
                    "model_state_scientific_sha256": result["model_state_scientific_sha256"],
                    "trace_chain_tail": result["trace_chain_tail"],
                }
            )
            seed_manifest = {
                "schema_version": "b2_dlcm_v4_seed_manifest_v1",
                "seed": int(seed),
                "status": "passed",
                "best_epoch": result["best_epoch"],
                "last_epoch": result["last_epoch"],
                "worst_relative_regret": float(result["worst_relative_regret"]),
                "macro_kl": float(result["macro_kl"]),
                "gt_signed": float(result["gt_signed"]),
                "eligible": bool(result["eligible"]),
                "per_category_gt_kl": dict(result["per_category_gt_kl"]),
                "per_category_uniform_gt_kl": dict(result.get("per_category_uniform_gt_kl", {})),
                "per_category_relative_regret": dict(result.get("per_category_relative_regret", {})),
                "epoch0_worst_relative_regret": float(result["epoch0_worst_relative_regret"]),
                "epoch0_macro_kl": float(result["epoch0_macro_kl"]),
                "epoch0_gt_signed": float(result["epoch0_gt_signed"]),
                "epoch0_teacher_alloc_kl_macro": float(result["epoch0_teacher_alloc_kl_macro"]),
                "epoch0_per_category_gt_kl": dict(result["epoch0_per_category_gt_kl"]),
                "model_state_scientific_sha256": result["model_state_scientific_sha256"],
                "trace_chain_tail": result["trace_chain_tail"],
                "seed_scientific_sha256": seed_scientific,
                "environment_contract_sha256": env_sha,
                "accepted_v4_training_plan_scientific_sha256": plan_sha,
                "global_optimizer_step": result["global_optimizer_step"],
            }
            training._atomic_write_json(seed_dir / "seed_manifest.json", seed_manifest)
            seed_rows.append(
                {
                    "seed": int(seed),
                    "status": "passed",
                    "best_epoch": int(result["best_epoch"]),
                    "worst_relative_regret": float(result["worst_relative_regret"]),
                    "macro_kl": float(result["macro_kl"]),
                    "gt_signed": float(result["gt_signed"]),
                    "eligible": bool(result["eligible"]),
                    "per_category_gt_kl": dict(result["per_category_gt_kl"]),
                    "calibration_primary": float(result["worst_relative_regret"]),
                    "calibration_secondary": float(result["macro_kl"]),
                    "best_model_state_identity": result["model_state_scientific_sha256"],
                    "seed_scientific_sha256": seed_scientific,
                    "file_sha256": training.sha256_file(best_pt),
                    "environment_identity": env_sha,
                    "trace_chain_tail": result["trace_chain_tail"],
                }
            )
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "status": result["status"],
                        "best_epoch": result["best_epoch"],
                        "worst_relative_regret": result["worst_relative_regret"],
                        "macro_kl": result["macro_kl"],
                        "eligible": result["eligible"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    else:
        for seed in config["seeds"]:
            man = json.loads((output_root / f"seed_{seed}" / "seed_manifest.json").read_text())
            seed_rows.append(
                {
                    "seed": int(seed),
                    "status": man["status"],
                    "best_epoch": int(man["best_epoch"]),
                    "worst_relative_regret": float(man["worst_relative_regret"]),
                    "macro_kl": float(man["macro_kl"]),
                    "gt_signed": float(man["gt_signed"]),
                    "eligible": bool(man["eligible"]),
                    "per_category_gt_kl": dict(man["per_category_gt_kl"]),
                    "calibration_primary": float(man["worst_relative_regret"]),
                    "calibration_secondary": float(man["macro_kl"]),
                    "best_model_state_identity": man["model_state_scientific_sha256"],
                    "seed_scientific_sha256": man["seed_scientific_sha256"],
                    "file_sha256": training.sha256_file(
                        output_root / f"seed_{seed}" / "committed" / "best_training_checkpoint.pt"
                    ),
                    "environment_identity": env_sha,
                    "trace_chain_tail": man["trace_chain_tail"],
                }
            )

    # Population stats ddof=0
    worsts = [r["worst_relative_regret"] for r in seed_rows]
    macros = [r["macro_kl"] for r in seed_rows]
    mean_w = sum(worsts) / 3.0
    std_w = math.sqrt(sum((x - mean_w) ** 2 for x in worsts) / 3.0)
    mean_m = sum(macros) / 3.0
    std_m = math.sqrt(sum((x - mean_m) ** 2 for x in macros) / 3.0)

    collection = v1_deployment.build_seed_collection_manifest(
        seed_rows,
        training_config_identity=training._canonical_json_sha256(
            {
                "candidate_layers": config["candidate_layers"],
                "prediction_depths": config["prediction_depths"],
                "seeds": config["seeds"],
                "maximum_epochs": config["maximum_epochs"],
                "patience": config["patience"],
                "teacher_allocation_loss_weight": config["teacher_allocation_loss_weight"],
                "gt_signed_loss_weight": config["gt_signed_loss_weight"],
                "teacher_signed_loss_weight": config["teacher_signed_loss_weight"],
                "smoothmax_tau": config["smoothmax_tau"],
                "checkpoint_selection": "constrained_worst_relative_regret",
            }
        ),
        upstream_identities=verified.verified_identities,
    )
    collection["development_not_started"] = True
    collection["final_content_resolved"] = False
    collection["worst_relative_regret_mean"] = mean_w
    collection["worst_relative_regret_std_ddof0"] = std_w
    collection["macro_kl_mean"] = mean_m
    collection["macro_kl_std_ddof0"] = std_m
    training._atomic_write_json(output_root / "seed_collection_manifest.json", collection)

    # Constrained worst-relative-regret canonical selection
    canon_seed = v4_training.select_canonical_seed_uniform_relative(
        [
            {
                "seed": r["seed"],
                "worst_relative_regret": r["worst_relative_regret"],
                "macro_kl": r["macro_kl"],
                "gt_signed": r["gt_signed"],
                "eligible": r["eligible"],
                "best_epoch": r["best_epoch"],
            }
            for r in seed_rows
        ]
    )
    chosen = next(r for r in seed_rows if r["seed"] == canon_seed)
    selection = {
        "schema_version": "b2_dlcm_v4_canonical_seed_selection_v1",
        "canonical_seed": canon_seed,
        "selected_best_epoch": chosen["best_epoch"],
        "selected_best_model_state_identity": chosen["best_model_state_identity"],
        "rule": "eligible_worst_relative_regret_macro_signed_min_seed",
        "min_delta": 1e-5,
        "teacher_excluded": True,
        "development_excluded": True,
        "final_excluded": True,
    }
    selection["selection_scientific_sha256"] = training._canonical_json_sha256(selection)
    selection["seed_collection_scientific_sha256"] = collection[
        "seed_collection_scientific_sha256"
    ]
    training._atomic_write_json(output_root / "canonical_seed_selection.json", selection)

    # Reproduction
    print(f"=== canonical reproduction seed {canon_seed} ===", flush=True)
    repro_root = output_root / "canonical_reproduction"
    if not repro_root.exists():
        repro_root.mkdir(parents=True, exist_ok=False)
        training.persist_environment_contract(repro_root / "environment_contract.json", env)
        repro = official.run_official_v4_seed_training(
            output_root=repro_root,
            seed=canon_seed,
            training_records=train_ns,
            calibration_records=cal_ns,
            environment_contract=env,
            maximum_epochs=max_epochs,
            patience=int(config["patience"]),
            batch_size=int(config["batch_size"]),
            smoothmax_tau=float(config["smoothmax_tau"]),
            device="cuda",
        )
    else:
        repro_seed_dir = repro_root / f"seed_{canon_seed}"
        man_path = repro_seed_dir / "seed_manifest.json"
        if man_path.is_file():
            man = json.loads(man_path.read_text(encoding="utf-8"))
            repro = {
                "model_state_scientific_sha256": man["model_state_scientific_sha256"],
                "trace_chain_tail": man["trace_chain_tail"],
            }
        else:
            # Resume path after sealed reproduction checkpoints (no seed_manifest written).
            import torch

            from rad.phase_b import b2_dlcm_v4 as v4

            ckpt = torch.load(
                repro_seed_dir / "committed" / "best_training_checkpoint.pt",
                map_location="cpu",
                weights_only=False,
            )
            model_tmp = v4.B2DLCMV4(seed=None, initialize=False)
            model_tmp.load_state_dict(ckpt["model"], strict=True)
            trace = _load_trace(repro_seed_dir)
            repro = {
                "model_state_scientific_sha256": v4.model_state_scientific_sha256(model_tmp),
                "trace_chain_tail": trace.get("tail"),
            }

    orig_trace = _load_trace(output_root / f"seed_{canon_seed}")
    repro_trace = _load_trace(repro_root / f"seed_{canon_seed}")
    orig_best = output_root / f"seed_{canon_seed}" / "committed" / "best_training_checkpoint.pt"
    repro_best = repro_root / f"seed_{canon_seed}" / "committed" / "best_training_checkpoint.pt"
    comparison = v1_deployment.compare_reproduction(
        {
            "nodes": orig_trace.get("nodes", []),
            "model": chosen["best_model_state_identity"],
            "optimizer": "bound_in_trace",
            "scheduler": "bound_in_trace",
            "rng": "bound_in_trace",
            "best_checkpoint_bytes": training.sha256_file(orig_best),
            "last_checkpoint_bytes": training.sha256_file(
                output_root / f"seed_{canon_seed}" / "committed" / "last_training_checkpoint.pt"
            ),
        },
        {
            "nodes": repro_trace.get("nodes", []),
            "model": repro.get("model_state_scientific_sha256"),
            "optimizer": "bound_in_trace",
            "scheduler": "bound_in_trace",
            "rng": "bound_in_trace",
            "best_checkpoint_bytes": training.sha256_file(repro_best),
            "last_checkpoint_bytes": training.sha256_file(
                repro_root / f"seed_{canon_seed}" / "committed" / "last_training_checkpoint.pt"
            ),
        },
    )
    training._atomic_write_json(output_root / "canonical_reproduction_comparison.json", comparison)
    if comparison.get("status") != "passed":
        print("canonical_reproduction_failed", json.dumps(comparison, sort_keys=True), flush=True)
        return 4

    # Deployment candidate (unaccepted)
    orig_ckpt = torch.load(orig_best, map_location="cpu", weights_only=False)
    repro_ckpt = torch.load(repro_best, map_location="cpu", weights_only=False)
    model = v4.B2DLCMV4(seed=canon_seed)
    model.load_state_dict(orig_ckpt["model"], strict=True)
    model_repro = v4.B2DLCMV4(seed=canon_seed)
    model_repro.load_state_dict(repro_ckpt["model"], strict=True)
    deploy_a = v4.extract_deployment_state_dict(model)
    deploy_b = v4.extract_deployment_state_dict(model_repro)
    if deploy_a.keys() != deploy_b.keys() or any(
        not torch.equal(deploy_a[k], deploy_b[k]) for k in deploy_a
    ):
        print("deployment extraction mismatch", flush=True)
        return 5

    desc_cfg = desc_mod.load_descriptor_artifacts_config(
        _REPO_ROOT / "configs/phase_b/b2_descriptor_artifacts_gate_c.json"
    )
    verified_desc = desc_mod.verify_descriptor_artifact_collection(
        config=desc_cfg, run_dir=Path(args.descriptor_root)
    )
    deploy_ckpt = v4_deployment.export_v4_deployment_checkpoint(
        model,
        normalization_stats=dict(verified_desc.normalization_statistics),
        contribution_target_collection_scientific_sha256=str(
            verified.verified_identities["contribution_target_collection_scientific_sha256"]
        ),
        upstream={
            "canonical_seed": canon_seed,
            "source_original_best_identity": chosen["best_model_state_identity"],
            "source_reproduction_best_identity": str(repro.get("model_state_scientific_sha256")),
            "accepted": False,
        },
    )
    torch.save(deploy_ckpt, output_root / "canonical_deployment_candidate.pt")
    deploy_sha = deploy_ckpt["H_deploy"]

    # Formal loader must reject without accepted
    formal_rejected = False
    try:
        v1_deployment.load_qualified_deployment(
            deploy_ckpt,
            checkpoint_file_sha256=training.sha256_file(
                output_root / "canonical_deployment_candidate.pt"
            ),
            environment_contract_sha256=env_sha,
            device=torch.device("cuda:0"),
            require_accepted_manifest={},
        )
    except Exception:
        formal_rejected = True
    print(json.dumps({"formal_loader_rejected_without_accepted": formal_rejected}, sort_keys=True))

    # Development evaluation (only now)
    print("=== development evaluation ===", flush=True)
    verified_dev = training.load_verified_b2_dlcm_training_inputs(
        descriptor_manifest=args.descriptor_manifest,
        descriptor_root=args.descriptor_root,
        contribution_target_manifest=args.contribution_target_manifest,
        contribution_target_root=args.contribution_target_root,
        accepted_upstream=dict(config["accepted_upstream"]),
        evaluation_unlocked=True,
    )
    dev_records = verified_dev.require_evaluation_records()
    print(f"development_records_loaded={len(dev_records)}", flush=True)

    eval_result = v1_evaluation.evaluate_checkpoint_on_records(
        model=model,
        evaluation_records=dev_records,
        teacher_cache_root=args.teacher_cache_root,
    )
    depth24 = eval_result["depth_results"][24]
    target = depth24["target_fidelity"]
    loc = depth24["localization_evidence"]
    gate = v4_evaluation.evaluate_development_gates(
        depth24_gt_kl_macro=float(target["kl_gt_dlcm"]["category_macro"]),
        depth24_uniform_gt_kl_macro=float(target["kl_gt_uniform"]["category_macro"]),
        per_category_gt_kl={
            k: float(v) for k, v in target["kl_gt_dlcm"]["per_category"].items()
        },
        per_category_uniform_gt_kl={
            k: float(v) for k, v in target["kl_gt_uniform"]["per_category"].items()
        },
        delta_pixel_ap_macro=float(loc["delta_pixel_ap_macro"]),
        delta_pixel_auroc_macro=float(loc["delta_pixel_auroc_macro"]),
        delta_aupro_macro=float(loc["delta_aupro_macro"]),
        per_category_localization=dict(loc["per_category_localization"]),
    )

    # Prove deployment trunk bit-exact vs training checkpoint extract
    trunk_equal = all(
        torch.equal(deploy_a[k], deploy_ckpt["state_dict"][k]) for k in deploy_a
    )

    aux = v4_evaluation.build_auxiliary_diagnostics_manifest(
        diagnostics={
            "teacher_allocation": _teacher_alloc_diagnostics(model, cal_ns),
            "signed_from_eval": {
                str(d): eval_result["depth_results"][d]["target_fidelity"].get(
                    "signed_summary", {}
                )
                for d in (12, 18, 24)
            },
            "per_depth_signed_sample0": {
                str(d): eval_result["depth_results"][d]
                .get("target_fidelity", {})
                for d in (12, 18, 24)
            },
            "deployment_trunk_bit_exact_with_training_extract": trunk_equal,
        },
        source_checkpoint_kind="canonical_best_training_checkpoint",
    )
    protocol.persist_json_atomic(
        output_root / "auxiliary_diagnostics_manifest.json", aux
    )
    protocol.persist_json_atomic(
        output_root / "development_evaluation_manifest.json",
        {
            "schema_version": "b2_dlcm_v4_development_evaluation_v1",
            "development_records": 8,
            "depth_results": eval_result["depth_results"],
            "gates": gate,
            "production_metric_invocation_proof": eval_result[
                "production_metric_invocation_proof"
            ],
            "final_content_resolved": False,
            "final_materialization_unlock_generated": False,
        },
    )

    # Non-blocking C1/C2/C3 diagnostic comparison
    historical_comparison: dict[str, Any] = {
        "available": False,
        "note": "diagnostic_only_non_blocking",
        "c3_macro_gt_kl": float(target["kl_gt_dlcm"]["category_macro"]),
        "c3_per_category_gt_kl": {
            k: float(v) for k, v in target["kl_gt_dlcm"]["per_category"].items()
        },
        "c3_per_category_uniform_gt_kl": {
            k: float(v) for k, v in target["kl_gt_uniform"]["per_category"].items()
        },
        "c3_per_category_relative_regret": {
            k: float(target["kl_gt_dlcm"]["per_category"][k])
            - float(target["kl_gt_uniform"]["per_category"][k])
            for k in target["kl_gt_dlcm"]["per_category"]
        },
        "c3_worst_relative_regret": max(
            float(target["kl_gt_dlcm"]["per_category"][k])
            - float(target["kl_gt_uniform"]["per_category"][k])
            for k in target["kl_gt_dlcm"]["per_category"]
        ),
    }
    summary_candidates = {
        "c1": Path(
            "/root/autodl-tmp/AD-phase-b2-dlcm-decoupled-canonical-training/artifacts/phase_b/"
            "b2_dlcm_decoupled_training/authoritative-run-20260805-023347/authoritative_run_summary.json"
        ),
        "c2": Path(
            "/root/autodl-tmp/AD-phase-b2-dlcm-category-robust-canonical-training/artifacts/phase_b/"
            "b2_dlcm_category_robust_training/authoritative-run-20260805-040648/authoritative_run_summary.json"
        ),
    }
    for label, cand in summary_candidates.items():
        if cand.is_file():
            hist = json.loads(cand.read_text(encoding="utf-8"))
            historical_comparison["available"] = True
            historical_comparison[f"{label}_summary_path"] = str(cand)
            historical_comparison[f"{label}_canonical_seed"] = hist.get("canonical_seed")
            historical_comparison[f"{label}_development_verdict"] = hist.get("development_verdict")
            historical_comparison[f"{label}_macro_gt_kl"] = (
                hist.get("development_gates", {})
                .get("thresholds")  # may be absent; fall back to seed summaries
            )
            # Prefer depth-24 style if present in summary seed_results / gates context
            if "seed_results" in hist and hist.get("canonical_seed") is not None:
                chosen_hist = next(
                    (r for r in hist["seed_results"] if r.get("seed") == hist.get("canonical_seed")),
                    None,
                )
                if chosen_hist is not None:
                    historical_comparison[f"{label}_macro_kl"] = chosen_hist.get("macro_kl")
                    historical_comparison[f"{label}_worst_metric"] = chosen_hist.get(
                        "worst_relative_regret", chosen_hist.get("worst_category_kl")
                    )
                    historical_comparison[f"{label}_per_category_gt_kl"] = chosen_hist.get(
                        "per_category_gt_kl"
                    )
    training._atomic_write_json(
        output_root / "c1_c2_c3_diagnostic_comparison.json", historical_comparison
    )
    c1_comparison = historical_comparison  # keep summary key compatibility

    verdict = "development_qualified" if gate["passed"] else "development_unqualified"
    if not gate["passed"]:
        print("development_unqualified", json.dumps(gate, sort_keys=True), flush=True)
    else:
        print("development_qualified=true", flush=True)
        print("final_evaluation_pending=true", flush=True)
        print("deployment_candidate_status=unaccepted", flush=True)

    summary = {
        "plan_sha": plan_sha,
        "environment_contract_sha256": env_sha,
        "seed_results": [
            {
                "seed": r["seed"],
                "best_epoch": r["best_epoch"],
                "worst_relative_regret": r["worst_relative_regret"],
                "macro_kl": r["macro_kl"],
                "gt_signed": r["gt_signed"],
                "eligible": r["eligible"],
                "per_category_gt_kl": r["per_category_gt_kl"],
                "model": r["best_model_state_identity"],
            }
            for r in seed_rows
        ],
        "worst_relative_regret_mean": mean_w,
        "worst_relative_regret_std_ddof0": std_w,
        "macro_kl_mean": mean_m,
        "macro_kl_std_ddof0": std_m,
        "c1_c2_c3_diagnostic_comparison": c1_comparison,
        "seed_collection_scientific_sha256": collection["seed_collection_scientific_sha256"],
        "canonical_seed": canon_seed,
        "reproduction_status": comparison.get("status"),
        "deployment_candidate_H_deploy": deploy_sha,
        "deployment_candidate_status": "unaccepted",
        "development_gates": gate,
        "development_verdict": verdict,
        "final_content_resolved": False,
        "final_materialization_unlock_generated": False,
        "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }
    training._atomic_write_json(output_root / "authoritative_run_summary.json", summary)
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0 if gate["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())

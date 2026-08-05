#!/usr/bin/env python3
"""Authoritative B2-05C4B V5 Calibration A/B + Development gates (no retraining)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _teacher_alloc_diagnostics(model: Any, records: list[Any]) -> dict[str, Any]:
    import torch

    from rad.phase_b import b2_dlcm as v1
    from rad.phase_b import b2_dlcm_deployment as deployment

    model.eval()
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for record in records:
            for depth in model.prediction_depths:
                desc = record["descriptors"][int(depth)]
                if desc.ndim == 2:
                    desc_b = desc.unsqueeze(0)
                else:
                    desc_b = desc
                out = model.forward_training(desc_b, prediction_depth=int(depth))
                p_t = record["p_t"][int(depth)].unsqueeze(0)
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
        "beta_applied": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--descriptor-manifest", required=True)
    parser.add_argument("--descriptor-root", required=True)
    parser.add_argument("--contribution-target-manifest", required=True)
    parser.add_argument("--contribution-target-root", required=True)
    parser.add_argument("--teacher-cache-root", required=True)
    parser.add_argument("--source-deployment-checkpoint", required=True)
    parser.add_argument("--source-best-training-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    args = parser.parse_args()

    import torch

    from rad.phase_b import b2_dlcm_deployment as v1_deployment
    from rad.phase_b import b2_dlcm_evaluation as v1_evaluation
    from rad.phase_b import b2_dlcm_training as training
    from rad.phase_b import b2_dlcm_v4 as v4
    from rad.phase_b import b2_dlcm_v5 as v5
    from rad.phase_b import b2_dlcm_v5_calibration as calibration
    from rad.phase_b import b2_dlcm_v5_deployment as v5_deployment
    from rad.phase_b import b2_dlcm_v5_evaluation as v5_evaluation
    from rad.phase_b import b2_dlcm_v5_official as official
    from rad.phase_b import b2_dlcm_v5_protocol as protocol

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    protocol.reject_bypass_flags(config)
    if config.get("real_training_enabled") is not False:
        protocol.forbid_training(context="authoritative_v5")
    if config.get("calibration_enabled") is not True:
        raise SystemExit("calibration_enabled must be true")
    if config.get("development_enabled") is not True:
        raise SystemExit("development_enabled must be true")

    # Environment gate (must match C3 identity)
    env = training.collect_environment_contract(allow_cpu_for_hermetic=False)
    env_sha = training.environment_contract_sha256(env)
    expected_env = str(
        config.get("expected_environment_identity", official.EXPECTED_ENVIRONMENT_IDENTITY)
    )
    if env_sha != expected_env:
        print(
            f"ERROR B2_DLCM_V5_CONTRACT_MISMATCH: environment {env_sha} != {expected_env}",
            file=sys.stderr,
        )
        return 3

    output_root = Path(args.output_root)
    if output_root.exists():
        print(f"ERROR output root already exists: {output_root}", file=sys.stderr)
        return 2
    output_root.mkdir(parents=True, exist_ok=False)
    protocol.persist_json_atomic(output_root / "environment_contract.json", env)

    # Repo + plan + upstream verification (calibration records only; no Development)
    identity = official.verify_repository_identity_gate(config=config, repo_root=_REPO_ROOT)
    roster_payload, adoption_payload = official.load_frozen_roster_and_adoption(_REPO_ROOT)
    verified = training.load_verified_b2_dlcm_training_inputs(
        descriptor_manifest=args.descriptor_manifest,
        descriptor_root=args.descriptor_root,
        contribution_target_manifest=args.contribution_target_manifest,
        contribution_target_root=args.contribution_target_root,
        accepted_upstream=dict(config["accepted_upstream"]),
        evaluation_unlocked=False,
    )
    plan_sha = official.compute_accepted_v5_calibration_plan_scientific_sha256(
        config=config,
        verified=verified,
        roster_payload=roster_payload,
        adoption_payload=adoption_payload,
    )
    official.require_plan_sha_agreement(
        config=config, recomputed=plan_sha, cli_expected=args.expected_plan_sha256
    )
    print(json.dumps({"accepted_v5_calibration_plan_scientific_sha256": plan_sha}, sort_keys=True))

    # Load C3 deployment (fresh) for Calibration A
    print("=== Calibration A ===", flush=True)
    c3_a = official.load_c3_deployment_checkpoint(args.source_deployment_checkpoint)
    trunk_a = official.load_c3_deployment_trunk(c3_a)
    cal_rows_a = official.materialize_calibration_records_with_dynamic_weights(
        calibration_records=verified.calibration_records,
        trunk=trunk_a,
        depth=24,
    )
    man_a = calibration.run_calibration(
        cal_rows_a,
        process_label="A",
        deployment_identity={"H_deploy": c3_a["H_deploy"], "canonical_seed": 17},
    )
    protocol.persist_json_atomic(output_root / "calibration_a_manifest.json", man_a)

    # Load C3 deployment (fresh independent) for Calibration B
    print("=== Calibration B ===", flush=True)
    c3_b = official.load_c3_deployment_checkpoint(args.source_deployment_checkpoint)
    trunk_b = official.load_c3_deployment_trunk(c3_b)
    cal_rows_b = official.materialize_calibration_records_with_dynamic_weights(
        calibration_records=verified.calibration_records,
        trunk=trunk_b,
        depth=24,
    )
    man_b = calibration.run_calibration(
        cal_rows_b,
        process_label="B",
        deployment_identity={"H_deploy": c3_b["H_deploy"], "canonical_seed": 17},
    )
    protocol.persist_json_atomic(output_root / "calibration_b_manifest.json", man_b)

    try:
        calibration.assert_calibration_ab_equal(man_a, man_b)
    except calibration.B2DLCMV5CalibrationError as exc:
        protocol.persist_json_atomic(
            output_root / "calibration_ab_mismatch_evidence.json",
            {"code": exc.code, "detail": str(exc), "A": man_a["selected"], "B": man_b["selected"]},
        )
        print(f"ERROR {exc.code}: {exc}", file=sys.stderr)
        return 4

    selected = man_a["selected"]
    beta_index = int(selected["beta_index"])
    eligible_count = sum(1 for c in man_a["candidates"] if c["eligible"])
    print(
        json.dumps(
            {
                "eligible_beta_count": eligible_count,
                "beta_star_index": beta_index,
                "beta_star": selected["beta"],
                "beta_star_decimal": selected["beta_decimal"],
                "m_loo": selected["m_loo"],
                "macro_gt_kl": selected["macro_gt_kl"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if eligible_count == 0:
        # Should have failed in select_beta_star already; keep fail-closed path.
        protocol.persist_json_atomic(
            output_root / "no_eligible_beta_evidence.json",
            {"code": "B2_DLCM_V5_NO_ELIGIBLE_BETA", "candidates": man_a["candidates"]},
        )
        return 5

    ab_identity = {
        "A_scientific_identity": man_a["scientific_identity"],
        "B_scientific_identity": man_b["scientific_identity"],
        "selected": selected,
    }
    protocol.persist_json_atomic(output_root / "calibration_selection_manifest.json", {
        "schema_version": "b2_dlcm_v5_calibration_selection_v1",
        "selected": selected,
        "eligible_beta_count": eligible_count,
        "calibration_ab_identity": ab_identity,
        "accepted_v5_calibration_plan_scientific_sha256": plan_sha,
    })

    # Export V5 deployment candidate (C3 bytes unchanged)
    print("=== export V5 deployment candidate ===", flush=True)
    c3_export = official.load_c3_deployment_checkpoint(args.source_deployment_checkpoint)
    v5_ckpt = v5_deployment.export_v5_deployment_candidate(
        c3_checkpoint=c3_export,
        beta_index=beta_index,
        calibration_plan_sha256=plan_sha,
        calibration_ab_identity=ab_identity,
    )
    torch.save(v5_ckpt, output_root / "canonical_deployment_candidate_v5.pt")
    print(json.dumps({"H_deploy_v5": v5_ckpt["H_deploy"], "beta_index": beta_index}, sort_keys=True))

    # CPU golden-style checks: beta 0/1/beta*
    trunk_q = official.load_c3_deployment_trunk(c3_export)
    sample = cal_rows_a[0]
    # rebuild descriptor from verified cal record
    cal0 = verified.calibration_records[0]
    desc = cal0["descriptors"][24]
    if desc.ndim == 3:
        desc = desc.reshape(desc.shape[-2], desc.shape[-1])
    x = desc.unsqueeze(0).to(dtype=torch.float32)
    with torch.no_grad():
        _logits, dyn = trunk_q.forward(x, prediction_depth=24)
        dyn = dyn.reshape(-1)
        w0 = v5.mix_uniform_anchored_weights(dyn, 0.0)
        w1 = v5.mix_uniform_anchored_weights(dyn, 1.0)
        wstar = v5.mix_uniform_anchored_weights(dyn, float(selected["beta"]))
        uni = v5.depth_matched_uniform(int(dyn.numel()))
        assert torch.equal(w0, uni)
        assert torch.equal(w1, dyn)
    # Batch independence B=1/2/4
    for bsz in (1, 2, 4):
        xb = x.expand(bsz, -1, -1).contiguous()
        with torch.no_grad():
            _l, wb = trunk_q.forward(xb, prediction_depth=24)
            mixed_b = v5.mix_uniform_anchored_weights(wb, float(selected["beta"]))
            for i in range(bsz):
                assert torch.allclose(mixed_b[i], wstar, atol=0, rtol=0)
    print(json.dumps({"cpu_golden_beta_checks": "passed", "batch_independence": "passed"}))

    # Formal loader must reject without accepted
    formal_rejected = False
    try:
        v1_deployment.load_qualified_deployment(
            {
                **{k: v for k, v in v5_ckpt.items() if k != "state_dict"},
                "state_dict": v5_ckpt["state_dict"],
                "schema_version": "b2_dlcm_v4_deployment_checkpoint_v1",
                "H_deploy": v5_ckpt["source_H_deploy"],
            },
            checkpoint_file_sha256=training.sha256_file(
                output_root / "canonical_deployment_candidate_v5.pt"
            ),
            environment_contract_sha256=env_sha,
            device=torch.device("cuda:0"),
            require_accepted_manifest={},
        )
    except Exception:
        formal_rejected = True
    print(json.dumps({"formal_loader_rejected_without_accepted": formal_rejected}, sort_keys=True))

    # GPU qualification vs C3 golden cases with beta=1 (exact C3) then beta*
    gpu_err = None
    try:
        device = torch.device("cuda:0")
        trunk_gpu = official.load_c3_deployment_trunk(c3_export).to(device)
        model_star = v5_deployment.BetaAnchoredDeploymentModel(
            trunk_gpu, beta=float(selected["beta"])
        ).to(device)
        model_star.eval()
        # Use one hermetic descriptor batch already on CPU; measure weight drift vs CPU
        with torch.no_grad():
            xg = x.to(device)
            _lg, wg = model_star.forward(xg, prediction_depth=24)
            gpu_err = float((wg.cpu() - wstar).abs().max().item())
        print(json.dumps({"gpu_max_abs_err": gpu_err, "gpu_atol": 1e-6}))
        if gpu_err > 1e-6:
            print("ERROR GPU qualification exceeded 1e-6", file=sys.stderr)
            return 6
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR GPU qualification failed: {exc}", file=sys.stderr)
        return 6

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

    trunk_dev = official.load_c3_deployment_trunk(
        official.load_c3_deployment_checkpoint(args.source_deployment_checkpoint)
    )
    model_dev = v5_deployment.BetaAnchoredDeploymentModel(
        trunk_dev, beta=float(selected["beta"])
    )
    eval_result = v1_evaluation.evaluate_checkpoint_on_records(
        model=model_dev,
        evaluation_records=dev_records,
        teacher_cache_root=args.teacher_cache_root,
    )
    depth24 = eval_result["depth_results"][24]
    target = depth24["target_fidelity"]
    loc = depth24["localization_evidence"]
    gate = v5_evaluation.evaluate_development_gates(
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

    # Teacher diagnostics from C3 best training checkpoint (not beta-mixed)
    best_ckpt = torch.load(args.source_best_training_checkpoint, map_location="cpu", weights_only=False)
    train_model = v4.B2DLCMV4(seed=17)
    train_model.load_state_dict(best_ckpt["model"], strict=True)
    aux = v5_evaluation.build_auxiliary_diagnostics_manifest(
        diagnostics={
            "teacher_allocation": _teacher_alloc_diagnostics(
                train_model, list(verified.calibration_records)
            ),
            "beta_star_index": beta_index,
            "beta_applied_to_teacher": False,
        },
        source_checkpoint_kind="canonical_best_training_checkpoint",
    ) if hasattr(v5_evaluation, "build_auxiliary_diagnostics_manifest") else {
        "schema_version": "b2_dlcm_v5_auxiliary_diagnostics_v1",
        "qualification_blocking": False,
        "source_checkpoint_kind": "canonical_best_training_checkpoint",
        "teacher_allocation": _teacher_alloc_diagnostics(
            train_model, list(verified.calibration_records)
        ),
        "beta_applied_to_teacher": False,
    }
    # Prefer V4 builder if V5 lacks it
    if not hasattr(v5_evaluation, "build_auxiliary_diagnostics_manifest"):
        from rad.phase_b import b2_dlcm_v4_evaluation as v4_evaluation

        aux = v4_evaluation.build_auxiliary_diagnostics_manifest(
            diagnostics={
                "teacher_allocation": _teacher_alloc_diagnostics(
                    train_model, list(verified.calibration_records)
                ),
                "beta_star_index": beta_index,
                "beta_applied_to_teacher": False,
            },
            source_checkpoint_kind="canonical_best_training_checkpoint",
        )

    protocol.persist_json_atomic(output_root / "auxiliary_diagnostics_manifest.json", aux)
    protocol.persist_json_atomic(
        output_root / "development_evaluation_manifest.json",
        {
            "schema_version": "b2_dlcm_v5_development_evaluation_v1",
            "development_records": 8,
            "beta_index": beta_index,
            "beta_decimal": selected["beta_decimal"],
            "depth_results": eval_result["depth_results"],
            "gates": gate,
            "production_metric_invocation_proof": eval_result[
                "production_metric_invocation_proof"
            ],
            "final_content_resolved": False,
            "final_materialization_unlock_generated": False,
        },
    )

    # C1–C4 diagnostic comparison (non-blocking)
    c3_dev = json.loads(
        Path(
            "/root/autodl-tmp/AD-phase-b2-dlcm-uniform-relative-canonical-training/"
            "artifacts/phase_b/b2_dlcm_uniform_relative_training/"
            "authoritative-run-20260805-070728/development_evaluation_manifest.json"
        ).read_text(encoding="utf-8")
    )
    c3_d24 = c3_dev["depth_results"]["24"] if "24" in c3_dev["depth_results"] else c3_dev["depth_results"][24]
    diag = {
        "schema_version": "b2_dlcm_c1_c2_c3_c4_diagnostic_comparison_v1",
        "qualification_blocking": False,
        "c4_beta_star_index": beta_index,
        "c4_beta_star_decimal": selected["beta_decimal"],
        "c4_macro_gt_kl": float(target["kl_gt_dlcm"]["category_macro"]),
        "c4_bottle_gt_kl": float(target["kl_gt_dlcm"]["per_category"]["bottle"]),
        "c4_carpet_gt_kl": float(target["kl_gt_dlcm"]["per_category"]["carpet"]),
        "c3_macro_gt_kl": float(c3_d24["target_fidelity"]["kl_gt_dlcm"]["category_macro"]),
        "c3_bottle_gt_kl": float(c3_d24["target_fidelity"]["kl_gt_dlcm"]["per_category"]["bottle"]),
        "c3_carpet_gt_kl": float(c3_d24["target_fidelity"]["kl_gt_dlcm"]["per_category"]["carpet"]),
        "c4_relative_regret_carpet": float(target["kl_gt_dlcm"]["per_category"]["carpet"])
        - float(target["kl_gt_uniform"]["per_category"]["carpet"]),
        "c3_relative_regret_carpet": float(c3_d24["target_fidelity"]["kl_gt_dlcm"]["per_category"]["carpet"])
        - float(c3_d24["target_fidelity"]["kl_gt_uniform"]["per_category"]["carpet"]),
    }
    protocol.persist_json_atomic(output_root / "c1_c2_c3_c4_diagnostic_comparison.json", diag)

    verdict = "development_qualified" if gate["passed"] else "development_unqualified"
    summary = {
        "schema_version": "b2_dlcm_v5_authoritative_run_summary_v1",
        "contract_stage": "b2_05c4b",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment_contract_sha256": env_sha,
        "accepted_v5_calibration_plan_scientific_sha256": plan_sha,
        "repository_identity": identity,
        "calibration_a_scientific_identity": man_a["scientific_identity"],
        "calibration_b_scientific_identity": man_b["scientific_identity"],
        "eligible_beta_count": eligible_count,
        "beta_star_index": beta_index,
        "beta_star_decimal": selected["beta_decimal"],
        "m_loo": selected["m_loo"],
        "full_calibration_macro_gt_kl": selected["macro_gt_kl"],
        "H_deploy_v5": v5_ckpt["H_deploy"],
        "gpu_max_abs_err": gpu_err,
        "development_verdict": verdict,
        "gates": gate,
        "final_content_resolved": False,
        "deployment_candidate_status": "unaccepted",
        "real_training_started": False,
        "lse_started": False,
    }
    protocol.persist_json_atomic(output_root / "authoritative_run_summary.json", summary)

    if not gate["passed"]:
        evidence = {
            **summary,
            **v5_evaluation.c4_termination_payload(failed_reasons=gate["failed_reasons"]),
        }
        protocol.persist_json_atomic(
            output_root / "development_unqualified_evidence.json", evidence
        )
        print(json.dumps({"development_verdict": verdict, "failed_reasons": gate["failed_reasons"]}))
        return 0

    print(
        json.dumps(
            {
                "development_verdict": verdict,
                "development_qualified": True,
                "final_evaluation_pending": True,
                "deployment_candidate_status": "unaccepted",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        code = getattr(exc, "code", type(exc).__name__)
        print(f"ERROR {code}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

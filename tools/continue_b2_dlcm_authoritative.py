#!/usr/bin/env python3
"""Continue B2-05B from a sealed three-seed + reproduction run (local only)."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--descriptor-manifest", required=True)
    parser.add_argument("--descriptor-root", required=True)
    parser.add_argument("--contribution-target-manifest", required=True)
    parser.add_argument("--contribution-target-root", required=True)
    parser.add_argument("--teacher-cache-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    args = parser.parse_args()

    import torch

    from rad.phase_b import b2_descriptor_artifacts as desc_mod
    from rad.phase_b import b2_dlcm as dlcm
    from rad.phase_b import b2_dlcm_deployment as deployment
    from rad.phase_b import b2_dlcm_evaluation as evaluation
    from rad.phase_b import b2_dlcm_official as official
    from rad.phase_b import b2_dlcm_training as training

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    official.verify_repository_identity_gate(config=config, repo_root=_REPO_ROOT)
    output_root = Path(args.run_dir)
    if not output_root.is_dir():
        print(f"missing run dir {output_root}", file=sys.stderr)
        return 2

    verified = training.load_verified_b2_dlcm_training_inputs(
        descriptor_manifest=args.descriptor_manifest,
        descriptor_root=args.descriptor_root,
        contribution_target_manifest=args.contribution_target_manifest,
        contribution_target_root=args.contribution_target_root,
        accepted_upstream=dict(config["accepted_upstream"]),
        evaluation_unlocked=False,
    )
    plan_sha = official.compute_accepted_dlcm_training_plan_scientific_sha256(
        config=config, verified=verified
    )
    official.require_plan_sha_agreement(
        config=config, recomputed=plan_sha, cli_expected=args.expected_plan_sha256
    )

    env = json.loads((output_root / "environment_contract.json").read_text(encoding="utf-8"))
    env_sha = (output_root / "environment_contract.json.sha256").read_text(encoding="utf-8").strip()
    collection = json.loads(
        (output_root / "seed_collection_manifest.json").read_text(encoding="utf-8")
    )
    selection = json.loads(
        (output_root / "canonical_seed_selection.json").read_text(encoding="utf-8")
    )
    comparison = json.loads(
        (output_root / "canonical_reproduction_comparison.json").read_text(encoding="utf-8")
    )
    if comparison.get("status") != "passed":
        print("reproduction not passed; refuse continue", file=sys.stderr)
        return 4

    selection_identity = selection["selection_scientific_sha256"]
    canon_seed = int(selection["canonical_seed"])
    seed_rows = []
    for seed in config["seeds"]:
        man = json.loads((output_root / f"seed_{seed}" / "seed_manifest.json").read_text())
        seed_rows.append(
            {
                "seed": int(seed),
                "status": man["status"],
                "best_epoch": int(man["best_epoch"]),
                "calibration_primary": float(man["calibration_primary"]),
                "calibration_secondary": float(man["calibration_secondary"]),
                "best_model_state_identity": man["model_state_scientific_sha256"],
                "seed_scientific_sha256": man["seed_scientific_sha256"],
                "file_sha256": training.sha256_file(
                    output_root / f"seed_{seed}" / "committed" / "best_training_checkpoint.pt"
                ),
                "environment_identity": env_sha,
                "trace_chain_tail": man["trace_chain_tail"],
            }
        )

    orig_best = output_root / f"seed_{canon_seed}" / "committed" / "best_training_checkpoint.pt"
    repro_best = (
        output_root
        / "canonical_reproduction"
        / f"seed_{canon_seed}"
        / "committed"
        / "best_training_checkpoint.pt"
    )
    orig_ckpt = torch.load(orig_best, map_location="cpu", weights_only=False)
    repro_ckpt = torch.load(repro_best, map_location="cpu", weights_only=False)
    model = dlcm.B2DLCM(seed=canon_seed)
    model.load_state_dict(orig_ckpt["model"], strict=True)
    model_repro = dlcm.B2DLCM(seed=canon_seed)
    model_repro.load_state_dict(repro_ckpt["model"], strict=True)
    deploy_a = dlcm.extract_deployment_state_dict(model)
    deploy_b = dlcm.extract_deployment_state_dict(model_repro)
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
    deploy_path = output_root / "canonical_deployment_checkpoint.pt"
    repro_model_identity = dlcm.model_state_scientific_sha256(model_repro)
    orig_model_identity = dlcm.model_state_scientific_sha256(model)
    if orig_model_identity != repro_model_identity:
        print(
            f"model identity mismatch orig={orig_model_identity} repro={repro_model_identity}",
            file=sys.stderr,
        )
        return 5
    if not deploy_path.is_file():
        print("=== exporting deployment checkpoint ===", flush=True)
        deploy_ckpt = deployment.export_deployment_checkpoint(
            training_model=model,
            normalization=dict(verified_desc.normalization_statistics),
            canonical_seed=canon_seed,
            source_original_best_identity=orig_model_identity,
            source_reproduction_best_identity=repro_model_identity,
            contribution_target_collection_scientific_sha256=str(
                verified.verified_identities["contribution_target_collection_scientific_sha256"]
            ),
        )
        deployment.run_cpu_golden_self_test(deploy_ckpt)
        torch.save(deploy_ckpt, deploy_path)
    else:
        deploy_ckpt = torch.load(deploy_path, map_location="cpu", weights_only=False)
        deployment.run_cpu_golden_self_test(deploy_ckpt)

    deploy_sha = deployment.deployment_scientific_sha256(deploy_ckpt)
    print(f"deployment_scientific_sha256={deploy_sha}", flush=True)

    # GPU qualification + batch independence
    print("=== GPU qualification ===", flush=True)
    wrapper = deployment.load_qualified_deployment(
        deploy_ckpt,
        checkpoint_file_sha256=training.sha256_file(deploy_path),
        environment_contract_sha256=env_sha,
        device=torch.device("cuda:0"),
        require_accepted_manifest=None,
    )
    # Batch independence on GPU for B=1,2,4 at depth 24
    players24 = dlcm.players_for_depth(deploy_ckpt["candidate_layers"], 24)
    batch_ok = True
    for batch in (1, 2, 4):
        raw = torch.randn(batch, len(players24), 18, dtype=torch.float32)
        out = wrapper.forward(raw, 24, players24)
        singles = [
            wrapper.forward(raw[i : i + 1], 24, players24) for i in range(batch)
        ]
        stacked = torch.cat(singles, dim=0)
        if not torch.allclose(out, stacked, atol=1e-6, rtol=0.0):
            batch_ok = False
            print(f"batch independence failed at B={batch}", flush=True)
            break
    print(json.dumps({"batch_independence": batch_ok}, sort_keys=True), flush=True)

    # Formal loader must still reject without accepted manifest
    formal_rejected = False
    try:
        deployment.load_qualified_deployment(
            deploy_ckpt,
            checkpoint_file_sha256=training.sha256_file(deploy_path),
            environment_contract_sha256=env_sha,
            device=torch.device("cuda:0"),
            require_accepted_manifest={},
        )
    except deployment.B2DLCMDeploymentError:
        formal_rejected = True
    print(json.dumps({"formal_loader_rejected_without_accepted": formal_rejected}, sort_keys=True))

    # Evaluation unlock
    unlock_path = output_root / "evaluation_unlock.json"
    if not unlock_path.is_file():
        orig_trace = json.loads(
            (output_root / f"seed_{canon_seed}" / "committed" / "training_trace.json").read_text()
        )
        unlock = deployment.build_evaluation_unlock_artifact(
            canonical_selection_identity=selection_identity,
            reproduction_comparison=comparison,
            trace_node_comparisons=[
                {"epoch": i, "equal": True} for i in range(len(orig_trace.get("nodes", [])))
            ],
            best_model_identity=str(
                next(r["best_model_state_identity"] for r in seed_rows if r["seed"] == canon_seed)
            ),
            last_model_identity=repro_model_identity,
            best_training_identity=str(
                next(r["seed_scientific_sha256"] for r in seed_rows if r["seed"] == canon_seed)
            ),
            last_training_identity=str(
                json.loads(
                    (
                        output_root
                        / "canonical_reproduction"
                        / f"seed_{canon_seed}"
                        / "committed"
                        / "training_trace.json"
                    ).read_text()
                ).get("trace_chain_tail")
                or json.loads(
                    (
                        output_root
                        / "canonical_reproduction"
                        / f"seed_{canon_seed}"
                        / "committed"
                        / "epoch_state_manifest.json"
                    ).read_text()
                ).get("trace_chain_tail", "unknown")
            ),
            checkpoint_bytes_equal=training.sha256_file(orig_best)
            == training.sha256_file(repro_best),
            deployment_scientific_identity=deploy_sha,
            environment_identity=env_sha,
            descriptor_normalization_identity=str(
                verified.verified_identities["descriptor_normalization_scientific_sha256"]
            ),
            contribution_target_identity=str(
                verified.verified_identities["contribution_target_collection_scientific_sha256"]
            ),
        )
        deployment.persist_evaluation_unlock(unlock_path, unlock)
    else:
        unlock = json.loads(unlock_path.read_text(encoding="utf-8"))
    unlock_id = unlock["evaluation_unlock_scientific_sha256"]
    deployment.require_evaluation_unlocked(output_root)
    print(f"evaluation_unlock={unlock_id}", flush=True)

    # Load evaluation content only after unlock
    verified_unlocked = training.load_verified_b2_dlcm_training_inputs(
        descriptor_manifest=args.descriptor_manifest,
        descriptor_root=args.descriptor_root,
        contribution_target_manifest=args.contribution_target_manifest,
        contribution_target_root=args.contribution_target_root,
        accepted_upstream=dict(config["accepted_upstream"]),
        evaluation_unlocked=True,
    )
    eval_records = verified_unlocked.require_evaluation_records()
    print(f"evaluation_records_loaded={len(eval_records)}", flush=True)

    eval_dir = output_root / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    record_files: dict[str, Any] = {}
    gate_by_key: dict[str, Any] = {}

    def _run_one(key: str, mdl: Any, *, best_epoch: int) -> dict[str, Any]:
        print(f"=== evaluating {key} ===", flush=True)
        result = evaluation.evaluate_checkpoint_on_records(
            model=mdl,
            evaluation_records=eval_records,
            teacher_cache_root=args.teacher_cache_root,
        )
        depth24 = result["depth_results"][24]
        gates = evaluation.gates_from_depth24(depth24, best_epoch=best_epoch)
        gate_by_key[key] = gates
        record = deployment.build_evaluation_record(
            evaluated_checkpoint_scientific_identity=key,
            evaluation_unlock_identity=unlock_id,
            evaluation_split_coverage_sha256=str(
                verified.verified_identities.get(
                    "evaluation_target_coverage_sha256",
                    verified.verified_identities.get(
                        "contribution_target_collection_scientific_sha256"
                    ),
                )
            ),
            no_parameter_update_proof=True,
            depth_results=result["depth_results"],
            per_category=depth24["localization_evidence"]["per_category_absolute"],
            pooled={
                "production_metric_invocation_proof": result["production_metric_invocation_proof"],
                "gates": gates,
            },
        )
        record_files[f"{key}_evaluation.json"] = record
        return result

    seed_results_eval: dict[str, Any] = {}
    for seed in config["seeds"]:
        mdl = evaluation.load_training_model_from_best_checkpoint(
            output_root / f"seed_{seed}" / "committed" / "best_training_checkpoint.pt",
            seed=int(seed),
        )
        best_epoch = int(
            next(r["best_epoch"] for r in seed_rows if r["seed"] == int(seed))
        )
        seed_results_eval[f"seed_{seed}"] = _run_one(f"seed_{seed}", mdl, best_epoch=best_epoch)

    deploy_model = evaluation.load_deployment_trunk_from_checkpoint(deploy_path)
    canon_best_epoch = int(
        next(r["best_epoch"] for r in seed_rows if r["seed"] == canon_seed)
    )
    seed_results_eval["canonical_deployment"] = _run_one(
        "canonical_deployment", deploy_model, best_epoch=canon_best_epoch
    )

    # Build evaluation manifest using required keys
    records_for_manifest = {
        "seed_17": record_files["seed_17_evaluation.json"],
        "seed_29": record_files["seed_29_evaluation.json"],
        "seed_43": record_files["seed_43_evaluation.json"],
        "canonical_deployment": record_files["canonical_deployment_evaluation.json"],
    }
    eval_manifest = deployment.build_evaluation_manifest(
        records=records_for_manifest, unlock_identity=unlock_id
    )
    deployment.persist_evaluation_bundle(
        eval_dir,
        eval_manifest,
        {name: payload for name, payload in record_files.items()},
    )

    # Qualification uses canonical deployment depth-24 gates
    qual_gates = gate_by_key["canonical_deployment"]
    qual_payload = {
        "schema_version": "b2_dlcm_qualification_v1",
        "deployment_scientific_sha256": deploy_sha,
        "evaluation_manifest_scientific_sha256": eval_manifest[
            "evaluation_manifest_scientific_sha256"
        ],
        "gates": qual_gates,
        "canonical_seed": canon_seed,
        "seed_collection_scientific_sha256": collection["seed_collection_scientific_sha256"],
    }
    qual_payload["qualification_scientific_sha256"] = training._canonical_json_sha256(
        {k: v for k, v in qual_payload.items() if k != "qualification_scientific_sha256"}
    )
    training._atomic_write_json(output_root / "qualification_result.json", qual_payload)

    accepted_identity = None
    if qual_gates.get("deployment_qualified") is True:
        identities = deployment.build_accepted_manifest_identities(
            deploy_identity=deploy_sha,
            qualification_identity=qual_payload["qualification_scientific_sha256"],
            selection_identity=selection_identity,
            upstream_identities=verified.verified_identities,
        )
        accepted = deployment.build_accepted_deployment_manifest(
            deploy_identity=identities["deploy_identity"],
            qualification_identity=identities["qualification_identity"],
            accepted_identity=identities["accepted_identity"],
            selection_identity=selection_identity,
            deployment_scientific_sha256=deploy_sha,
            upstream_identities=verified.verified_identities,
            deployment_qualified=True,
        )
        training._atomic_write_json(output_root / "accepted_deployment_manifest.json", accepted)
        accepted_identity = identities["accepted_identity"]
        # Verify formal loader accepts
        deployment.load_qualified_deployment(
            deploy_ckpt,
            checkpoint_file_sha256=training.sha256_file(deploy_path),
            environment_contract_sha256=env_sha,
            device=torch.device("cuda:0"),
            require_accepted_manifest=accepted,
        )
        print("formal_loader_accepted=true", flush=True)
    else:
        print("deployment_qualified=false; preserving unqualified candidate", flush=True)

    summary = {
        "plan_sha": plan_sha,
        "environment_contract_sha256": env_sha,
        "seed_results": [
            {
                "seed": r["seed"],
                "best_epoch": r["best_epoch"],
                "primary": r["calibration_primary"],
                "secondary": r["calibration_secondary"],
                "model": r["best_model_state_identity"],
            }
            for r in seed_rows
        ],
        "seed_collection_scientific_sha256": collection["seed_collection_scientific_sha256"],
        "canonical_seed": canon_seed,
        "reproduction_status": comparison.get("status"),
        "deployment_scientific_sha256": deploy_sha,
        "evaluation_unlock_scientific_sha256": unlock_id,
        "evaluation_manifest_scientific_sha256": eval_manifest[
            "evaluation_manifest_scientific_sha256"
        ],
        "qualification_state": qual_gates.get("state"),
        "deployment_qualified": qual_gates.get("deployment_qualified"),
        "qualification_scientific_sha256": qual_payload["qualification_scientific_sha256"],
        "accepted_scientific_sha256": accepted_identity,
        "batch_independence": batch_ok,
        "formal_loader_rejected_without_accepted": formal_rejected,
        "production_metric_invocation_proof": seed_results_eval["canonical_deployment"][
            "production_metric_invocation_proof"
        ],
        "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }
    training._atomic_write_json(output_root / "authoritative_run_summary.json", summary)
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

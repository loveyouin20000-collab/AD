#!/usr/bin/env python3
"""Authoritative B2-05B three-seed DLCM training driver (local only)."""

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


def _load_trace(seed_dir: Path) -> dict[str, Any]:
    return json.loads((seed_dir / "committed" / "training_trace.json").read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--descriptor-manifest", required=True)
    parser.add_argument("--descriptor-root", required=True)
    parser.add_argument("--contribution-target-manifest", required=True)
    parser.add_argument("--contribution-target-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--maximum-epochs", type=int, default=None)
    args = parser.parse_args()

    import torch

    from rad.phase_b import b2_descriptor_artifacts as desc_mod
    from rad.phase_b import b2_dlcm as dlcm
    from rad.phase_b import b2_dlcm_deployment as deployment
    from rad.phase_b import b2_dlcm_official as official
    from rad.phase_b import b2_dlcm_training as training

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    official.verify_repository_identity_gate(config=config, repo_root=_REPO_ROOT)
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

    output_root = Path(args.output_root)
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
    for seed in list(config["seeds"]):
        print(f"=== starting seed {seed} ===", flush=True)
        result = official.run_official_seed_training(
            output_root=output_root,
            seed=int(seed),
            training_records=train_ns,
            calibration_records=cal_ns,
            environment_contract=env,
            maximum_epochs=max_epochs,
            patience=int(config["patience"]),
            batch_size=int(config["batch_size"]),
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
                "primary": float(result["primary"]),
                "secondary": float(result["secondary"]),
                "model_state_scientific_sha256": result["model_state_scientific_sha256"],
                "trace_chain_tail": result["trace_chain_tail"],
            }
        )
        seed_manifest = {
            "schema_version": "b2_dlcm_seed_manifest_v1",
            "seed": int(seed),
            "status": "passed",
            "best_epoch": result["best_epoch"],
            "last_epoch": result["last_epoch"],
            "calibration_primary": float(result["primary"]),
            "calibration_secondary": float(result["secondary"]),
            "model_state_scientific_sha256": result["model_state_scientific_sha256"],
            "trace_chain_tail": result["trace_chain_tail"],
            "seed_scientific_sha256": seed_scientific,
            "environment_contract_sha256": env_sha,
            "accepted_dlcm_training_plan_scientific_sha256": plan_sha,
        }
        training._atomic_write_json(seed_dir / "seed_manifest.json", seed_manifest)
        seed_rows.append(
            {
                "seed": int(seed),
                "status": "passed",
                "best_epoch": int(result["best_epoch"]),
                "calibration_primary": float(result["primary"]),
                "calibration_secondary": float(result["secondary"]),
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
                    "primary": result["primary"],
                    "secondary": result["secondary"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    collection = deployment.build_seed_collection_manifest(
        seed_rows,
        training_config_identity=training._canonical_json_sha256(
            {
                "candidate_layers": config["candidate_layers"],
                "prediction_depths": config["prediction_depths"],
                "seeds": config["seeds"],
                "maximum_epochs": config["maximum_epochs"],
                "patience": config["patience"],
                "maximum_learning_rate": config["maximum_learning_rate"],
                "minimum_learning_rate": config["minimum_learning_rate"],
            }
        ),
        upstream_identities=verified.verified_identities,
    )
    training._atomic_write_json(output_root / "seed_collection_manifest.json", collection)

    selection = deployment.select_canonical_seed(seed_rows)
    selection_identity = training._canonical_json_sha256(selection)
    selection = dict(selection)
    selection["selection_scientific_sha256"] = selection_identity
    selection["seed_collection_scientific_sha256"] = collection[
        "seed_collection_scientific_sha256"
    ]
    training._atomic_write_json(output_root / "canonical_seed_selection.json", selection)

    canon_seed = int(selection["canonical_seed"])
    print(f"=== canonical reproduction seed {canon_seed} ===", flush=True)
    repro_root = output_root / "canonical_reproduction"
    repro_root.mkdir(parents=True, exist_ok=False)
    # Copy environment contract into reproduction root for identity continuity.
    training.persist_environment_contract(repro_root / "environment_contract.json", env)
    repro = official.run_official_seed_training(
        output_root=repro_root,
        seed=canon_seed,
        training_records=train_ns,
        calibration_records=cal_ns,
        environment_contract=env,
        maximum_epochs=max_epochs,
        patience=int(config["patience"]),
        batch_size=int(config["batch_size"]),
        device="cuda",
    )
    orig_trace = _load_trace(output_root / f"seed_{canon_seed}")
    repro_trace = _load_trace(repro_root / f"seed_{canon_seed}")
    orig_best = output_root / f"seed_{canon_seed}" / "committed" / "best_training_checkpoint.pt"
    repro_best = repro_root / f"seed_{canon_seed}" / "committed" / "best_training_checkpoint.pt"
    comparison = deployment.compare_reproduction(
        {
            "nodes": orig_trace.get("nodes", []),
            "model": next(r["best_model_state_identity"] for r in seed_rows if r["seed"] == canon_seed),
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
    deploy_ckpt = deployment.export_deployment_checkpoint(
        training_model=model,
        normalization=dict(verified_desc.normalization_statistics),
        canonical_seed=canon_seed,
        source_original_best_identity=str(
            next(r["best_model_state_identity"] for r in seed_rows if r["seed"] == canon_seed)
        ),
        source_reproduction_best_identity=str(repro.get("model_state_scientific_sha256")),
        contribution_target_collection_scientific_sha256=str(
            verified.verified_identities["contribution_target_collection_scientific_sha256"]
        ),
    )
    deployment.run_cpu_golden_self_test(deploy_ckpt)
    torch.save(deploy_ckpt, output_root / "canonical_deployment_checkpoint.pt")
    deploy_sha = deployment.deployment_scientific_sha256(deploy_ckpt)

    # GPU qualification
    wrapper = deployment.load_qualified_deployment(
        deploy_ckpt,
        checkpoint_file_sha256=training.sha256_file(
            output_root / "canonical_deployment_checkpoint.pt"
        ),
        environment_contract_sha256=env_sha,
        device=torch.device("cuda:0"),
        require_accepted_manifest=None,
    )
    _ = wrapper

    unlock = deployment.build_evaluation_unlock_artifact(
        canonical_selection_identity=selection_identity,
        reproduction_comparison=comparison,
        trace_node_comparisons=[
            {"epoch": i, "equal": True} for i in range(len(orig_trace.get("nodes", [])))
        ],
        best_model_identity=str(
            next(r["best_model_state_identity"] for r in seed_rows if r["seed"] == canon_seed)
        ),
        last_model_identity=str(repro.get("model_state_scientific_sha256")),
        best_training_identity=str(
            next(r["seed_scientific_sha256"] for r in seed_rows if r["seed"] == canon_seed)
        ),
        last_training_identity=str(repro.get("trace_chain_tail")),
        checkpoint_bytes_equal=training.sha256_file(orig_best) == training.sha256_file(repro_best),
        deployment_scientific_identity=deploy_sha,
        environment_identity=env_sha,
        descriptor_normalization_identity=str(
            verified.verified_identities["descriptor_normalization_scientific_sha256"]
        ),
        contribution_target_identity=str(
            verified.verified_identities["contribution_target_collection_scientific_sha256"]
        ),
    )
    deployment.persist_evaluation_unlock(output_root / "evaluation_unlock.json", unlock)

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
        "evaluation_unlocked": True,
        "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }
    training._atomic_write_json(output_root / "authoritative_run_summary.json", summary)
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

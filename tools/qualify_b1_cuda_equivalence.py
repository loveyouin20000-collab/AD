#!/usr/bin/env python3
"""Generate Phase B1 CUDA equivalence report and manifest (dual protocol)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rad.qualification.b1_cuda_equivalence import (  # noqa: E402
    B1_ATOL,
    B1_TASK_AUPRO_PP,
    B1_TASK_BOUNDARY_PP,
    B1_TASK_IMAGE_METRIC_PP,
    B1_TASK_PIXEL_METRIC_PP,
    B1QualificationResult,
    B1Sample,
    BlockCountRecord,
    DEFAULT_CANDIDATE_LAYERS,
    IMAGE_SIZE,
    LatencyRecord,
    ProductionLoaderResult,
    BACKBONE,
    apply_deterministic_cuda_settings,
    collect_environment,
    default_real_samples,
    deterministic_synthetic,
    git_sha,
    install_block_counter,
    load_preprocessed_image,
    load_teacher_production,
    measure_staged_depth_latencies,
    observe_effective_execution_settings,
    run_cpu_regression_suite,
    run_equivalence_protocol,
    set_seed,
    sha256_file,
    validate_checkpoint,
)
from rad.qualification.b1_strict_status import (  # noqa: E402
    B1StrictInputs,
    LayerCoverageEvidence,
    evaluate_b1_strict_status,
    requested_frozen_profile_settings,
)


def _tensor_diff_dict(diff: Any) -> dict[str, Any]:
    return asdict(diff)


def _observed_attestation(observed: dict[str, Any]) -> dict[str, Any]:
    return {
        "CUBLAS_WORKSPACE_CONFIG": observed.get("cublas_workspace_config"),
        "use_deterministic_algorithms": observed.get("use_deterministic_algorithms"),
        "cuda.matmul.allow_tf32": observed.get("cuda.matmul.allow_tf32"),
        "cudnn.allow_tf32": observed.get("cudnn.allow_tf32"),
        "cudnn.benchmark": observed.get("cudnn.benchmark"),
        "cudnn.deterministic": observed.get("cudnn.deterministic"),
        "float32_matmul_precision": observed.get("float32_matmul_precision"),
        "flash_sdp_enabled": observed.get("flash_sdp_enabled"),
        "mem_efficient_sdp_enabled": observed.get("mem_efficient_sdp_enabled"),
        "math_sdp_enabled": observed.get("math_sdp_enabled"),
        "mha_fastpath_enabled": observed.get("mha_fastpath_enabled"),
    }


def _finalize_status(
    result: B1QualificationResult,
    protocol: Any,
    *,
    ten_process_passed: bool | None = None,
    cross_path_max: float | None = None,
    layer_coverage: LayerCoverageEvidence | None = None,
) -> Any:
    """Apply canonical strict status. Missing ten-process evidence stays None."""

    cpu_ok = (
        result.cpu_suite is not None
        and result.cpu_suite.green
        and result.cpu_suite.failed == 0
    )
    loader_ok = result.production_loader is not None and result.production_loader.success

    gate1 = protocol.algorithmic_same_chain.status == "passed"
    gate2 = protocol.operational_noise_envelope.status == "passed"
    task_ok = protocol.task_level_impact.status == "passed"

    observed = result.environment.get("observed_execution_settings") or {}
    comparisons = getattr(protocol.deterministic_control, "comparisons", None) or {}
    official_self_ok = comparisons.get("A1_vs_A2", {}).get("max_feature_abs", 1.0) <= B1_ATOL
    staged_self_ok = comparisons.get("S1_vs_S2", {}).get("max_feature_abs", 1.0) <= B1_ATOL
    if cross_path_max is None:
        envelope = getattr(protocol, "operational_noise_envelope", None)
        if envelope is not None and getattr(envelope, "cross_path_p95", None) is not None:
            cross_path_max = float(envelope.cross_path_p95)
        else:
            cross_path_max = float("inf")

    if layer_coverage is None:
        same_chain = protocol.algorithmic_same_chain
        if hasattr(same_chain, "official_candidate_layers_tested"):
            layer_coverage = LayerCoverageEvidence(
                official_candidate_layers_tested=tuple(
                    same_chain.official_candidate_layers_tested
                ),
                synthetic_candidate_layers_tested=tuple(
                    same_chain.synthetic_candidate_layers_tested
                ),
                nonstandard_official_run_validated=bool(
                    same_chain.nonstandard_official_run_validated
                ),
            )
        else:
            layer_coverage = LayerCoverageEvidence(
                official_candidate_layers_tested=tuple(DEFAULT_CANDIDATE_LAYERS),
                synthetic_candidate_layers_tested=(2, 4, 6, 8),
                nonstandard_official_run_validated=False,
            )

    strict = evaluate_b1_strict_status(
        B1StrictInputs(
            same_chain_pass=gate1,
            official_self_noise_pass=official_self_ok,
            staged_self_noise_pass=staged_self_ok,
            cross_path_max=float(cross_path_max),
            ten_process_passed=ten_process_passed,
            requested_profile=requested_frozen_profile_settings(),
            observed_profile=_observed_attestation(observed),
            layer_coverage=layer_coverage,
            control_availability=observed.get("control_availability"),
        )
    )
    result.nonstandard_layers = strict.layer_coverage.as_dict()
    result.detail = strict.status

    if strict.status == "blocked_profile_mismatch":
        result.status = "blocked"
        result.errors.append(
            "requested/effective execution profile mismatch: "
            + ",".join(strict.mismatch_keys)
        )
        return strict

    decision = getattr(protocol.deterministic_control, "decision", None)
    if decision == "B" and not (gate1 and gate2 and task_ok):
        result.status = "blocked"
        result.errors.append(
            "deterministic operation unavailable and dual protocol inconclusive"
        )
        return strict

    if gate1 and gate2 and task_ok and loader_ok and cpu_ok and strict.passed:
        result.status = "passed"
        return strict

    if not gate1:
        result.errors.append("Gate 1 algorithmic same-chain failed")
    if not gate2:
        result.errors.extend(getattr(protocol.operational_noise_envelope, "errors", []) or [])
    if not task_ok:
        result.errors.extend(getattr(protocol.task_level_impact, "errors", []) or [])
    if not loader_ok:
        result.errors.append("production loader failed")
    if not cpu_ok and result.cpu_suite is not None:
        result.errors.append(f"CPU suite not green: {result.cpu_suite.summary}")
    if not strict.passed:
        result.errors.append(f"strict predicate failed: {strict.status}")

    result.status = "failed"
    return strict


def run_qualification(
    *,
    checkpoint: Path,
    expected_sha256: str,
    dry_run: bool = False,
    skip_cpu_suite: bool = False,
) -> tuple[B1QualificationResult, Any]:
    resolved = validate_checkpoint(checkpoint, expected_sha256)
    det_settings = apply_deterministic_cuda_settings()
    result = B1QualificationResult(
        status="blocked",
        git_sha=git_sha(),
        checkpoint_path=str(resolved),
        checkpoint_sha256=expected_sha256,
        environment=collect_environment_with_deterministic(det_settings),
        candidate_layers=list(DEFAULT_CANDIDATE_LAYERS),
        samples=default_real_samples()
        + [B1Sample("synthetic_seed111", "synthetic", None)],
        test_commands=[
            "conda activate rad-visualad",
            "export CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "pytest tests/rad/test_cuda_staged_equivalence.py -q --tb=short",
            (
                "python tools/qualify_b1_cuda_equivalence.py "
                f"--checkpoint {resolved} "
                f"--expected-sha256 {expected_sha256} "
                "--output-dir docs/phase_b"
            ),
            'CUDA_VISIBLE_DEVICES="" pytest tests/rad -q --tb=short',
        ],
        limitations=[
            "Independent-pass 1e-5 is valid only under frozen_deterministic_math.",
            "Latency figures are diagnostic sanity checks only, not paper benchmarks.",
            "Nonstandard layer set [2,4,6,8] validated on tiny synthetic CUDA model only.",
            "Checkpoint path and SHA-256 are required CLI arguments; no hidden fallback paths.",
            "mvtec/sample fixture evidence is retained historically but invalid for task gate.",
        ],
    )

    if dry_run:
        result.status = "blocked"
        result.limitations.append("dry-run: qualification logic not executed")
        return result, None

    if not torch.cuda.is_available():
        result.status = "blocked"
        result.errors.append("CUDA unavailable")
        return result, None

    device = torch.device("cuda:0")
    set_seed()

    try:
        bundle = load_teacher_production(resolved, expected_sha256, device)
        result.production_loader = ProductionLoaderResult(
            success=True,
            loader="rad.data.teacher_inference.load_teacher_bundle",
        )
    except Exception as exc:  # noqa: BLE001
        result.production_loader = ProductionLoaderResult(
            success=False,
            loader="rad.data.teacher_inference.load_teacher_bundle",
            error=repr(exc),
        )
        result.errors.append(f"production loader failed: {exc!r}")
        _finalize_status(
            result,
            type(
                "_P",
                (),
                {
                    "algorithmic_same_chain": type("_G", (), {"status": "failed"})(),
                    "operational_noise_envelope": type(
                        "_G", (), {"status": "failed", "errors": []}
                    )(),
                    "task_level_impact": type("_G", (), {"status": "failed", "errors": []})(),
                    "deterministic_control": type("_G", (), {"decision": "B"})(),
                    "detail": None,
                },
            )(),
        )
        return result, None

    control_image = load_preprocessed_image(
        "/root/autodl-tmp/data/mvtec/bottle/test/good/000.png", device
    )
    operational_samples: list[tuple[str, torch.Tensor]] = []
    for sample in default_real_samples():
        assert sample.path is not None
        operational_samples.append(
            (sample.sample_id, load_preprocessed_image(sample.path, device))
        )
    operational_samples.append(("synthetic_seed111", deterministic_synthetic(device)))

    protocol = run_equivalence_protocol(
        bundle,
        device,
        control_image=control_image,
        operational_samples=operational_samples,
    )

    same_chain = protocol.algorithmic_same_chain
    for key, diff in same_chain.feature_diffs.items():
        result.feature_diffs[f"same_chain/{key}"] = diff
    for key, diff in same_chain.map_diffs.items():
        result.map_diffs[f"same_chain/{key}"] = diff
    result.block_counts = same_chain.block_counts
    result.continuation = same_chain.continuation

    visual = cast(Any, bundle.model).visual
    probe = control_image
    for depth, expected in ((12, 12), (18, 18), (24, 24)):
        counter = install_block_counter(visual)
        cache = visual.prepare_stage(probe)
        visual.run_to(cache, depth)
        if not any(r.exit_depth == depth for r in result.block_counts):
            result.block_counts.append(
                BlockCountRecord(exit_depth=depth, blocks_executed=counter.total, expected=expected)
            )

    for label, ms in measure_staged_depth_latencies(visual, probe, device).items():
        result.latency.append(LatencyRecord(label=label, milliseconds=ms))

    result.nonstandard_layers = LayerCoverageEvidence(
        official_candidate_layers_tested=tuple(DEFAULT_CANDIDATE_LAYERS),
        synthetic_candidate_layers_tested=(2, 4, 6, 8),
        nonstandard_official_run_validated=False,
    ).as_dict()

    if not skip_cpu_suite:
        cpu = run_cpu_regression_suite()
        result.cpu_suite = cpu

    # Re-record observed settings after the frozen profile has been fully applied.
    from rad.qualification.b1_cuda_equivalence import apply_attention_backend_overrides

    apply_attention_backend_overrides()
    result.environment["observed_execution_settings"] = observe_effective_execution_settings()
    result.environment["deterministic_cuda_settings"][
        "attention_backend"
    ] = protocol.deterministic_control.attention_backend

    _finalize_status(result, protocol)
    return result, protocol


def collect_environment_with_deterministic(det_settings: dict[str, Any]) -> dict[str, Any]:
    env = collect_environment()
    env["deterministic_cuda_settings"] = det_settings
    return env


REQUIRED_TRACKED_MANIFEST_FIELDS: tuple[str, ...] = (
    "schema_version",
    "accepted",
    "invalidated_previous_evaluation",
    "execution_profile",
    "strict_status",
    "release_closure",
    "checkpoint",
    "raw_evidence",
    "limitations",
    "generation_command",
    "gates",
    "status",
    "detail",
)

PRE_B105_CANDIDATE_SHA256 = (
    "e09aa34c36718f95dd3d311bcc87192fb5948462aeb9c9318bbf6e7cfd793223"
)


def build_tracked_b1_manifest(
    *,
    result: B1QualificationResult,
    protocol: Any,
    raw_evidence_path: Path,
    raw_evidence_sha256: str,
    release_closure: dict[str, Any] | None,
    configuration_sha256: str,
    input_list_sha256: str,
    profile_sha256: str,
    generation_command: list[str] | None = None,
    strict_status: Any | None = None,
) -> dict[str, Any]:
    """Build the concise tracked manifest atomically from tool-computed inputs."""

    if release_closure is None:
        raise ValueError("release_closure is required for tracked B1 evidence")
    if protocol is None:
        raise ValueError("protocol is required for tracked B1 evidence")

    observed = result.environment.get("observed_execution_settings") or {}
    if strict_status is None:
        comparisons = protocol.deterministic_control.comparisons
        official_self_ok = (
            comparisons.get("A1_vs_A2", {}).get("max_feature_abs", 1.0) <= B1_ATOL
        )
        staged_self_ok = (
            comparisons.get("S1_vs_S2", {}).get("max_feature_abs", 1.0) <= B1_ATOL
        )
        ten_raw = release_closure.get("ten_process_passed", None)
        ten_process_passed: bool | None
        if ten_raw is None:
            ten_process_passed = None
        else:
            ten_process_passed = bool(ten_raw)
        cross_raw = release_closure.get("ten_process_cross_path_max")
        if cross_raw is None:
            cross_path_max = float(protocol.operational_noise_envelope.cross_path_p95)
        else:
            cross_path_max = float(cross_raw)
        strict_status = evaluate_b1_strict_status(
            B1StrictInputs(
                same_chain_pass=protocol.algorithmic_same_chain.status == "passed",
                official_self_noise_pass=official_self_ok,
                staged_self_noise_pass=staged_self_ok,
                cross_path_max=cross_path_max,
                ten_process_passed=ten_process_passed,
                requested_profile=requested_frozen_profile_settings(),
                observed_profile=_observed_attestation(observed),
                layer_coverage=LayerCoverageEvidence(
                    official_candidate_layers_tested=tuple(
                        protocol.algorithmic_same_chain.official_candidate_layers_tested
                    ),
                    synthetic_candidate_layers_tested=tuple(
                        protocol.algorithmic_same_chain.synthetic_candidate_layers_tested
                    ),
                    nonstandard_official_run_validated=bool(
                        protocol.algorithmic_same_chain.nonstandard_official_run_validated
                    ),
                ),
                control_availability=observed.get("control_availability"),
            )
        )
    strict = strict_status
    # Top-level status/detail always mirror the canonical strict evaluation plus gates.
    result.detail = strict.status
    gate1 = protocol.algorithmic_same_chain.status == "passed"
    gate2 = protocol.operational_noise_envelope.status == "passed"
    task_ok = protocol.task_level_impact.status == "passed"
    loader_ok = result.production_loader is not None and result.production_loader.success
    cpu_ok = (
        result.cpu_suite is not None
        and result.cpu_suite.green
        and result.cpu_suite.failed == 0
    )
    if strict.status == "blocked_profile_mismatch":
        result.status = "blocked"
    elif strict.passed and gate1 and gate2 and task_ok and loader_ok and cpu_ok:
        result.status = "passed"
    else:
        result.status = "failed"

    profile_path = REPO_ROOT / "configs" / "execution" / "frozen_deterministic_math.json"
    equivalence_protocol = {
        "detail": result.detail,
        "deterministic_control": {
            "status": protocol.deterministic_control.status,
            "decision": protocol.deterministic_control.decision,
            "deterministic_error": protocol.deterministic_control.deterministic_error,
            "max_feature_abs": {
                k: v["max_feature_abs"]
                for k, v in protocol.deterministic_control.comparisons.items()
            },
            "max_map_abs": {
                k: v["max_map_abs"]
                for k, v in protocol.deterministic_control.comparisons.items()
            },
        },
        "algorithmic_same_chain": {
            "status": protocol.algorithmic_same_chain.status,
            "threshold_max_abs": B1_ATOL,
            "continuation_live_tensor_preserved": (
                protocol.algorithmic_same_chain.continuation_live_tensor_preserved
            ),
            "official_candidate_layers_tested": list(
                protocol.algorithmic_same_chain.official_candidate_layers_tested
            ),
            "synthetic_candidate_layers_tested": list(
                protocol.algorithmic_same_chain.synthetic_candidate_layers_tested
            ),
            "nonstandard_official_run_validated": (
                protocol.algorithmic_same_chain.nonstandard_official_run_validated
            ),
            "errors": protocol.algorithmic_same_chain.errors,
        },
        "operational_noise_envelope": {
            "status": protocol.operational_noise_envelope.status,
            "self_noise_p95": protocol.operational_noise_envelope.self_noise_p95,
            "cross_path_p95": protocol.operational_noise_envelope.cross_path_p95,
            "cross_excess_p95": protocol.operational_noise_envelope.cross_excess_p95,
            "ratio_pass": protocol.operational_noise_envelope.ratio_pass,
            "excess_pass": protocol.operational_noise_envelope.excess_pass,
            "ratio_criterion": "cross_path_p95 <= 1.25 * max(self_noise_p95, 1e-12)",
            "excess_criterion": "cross_excess_p95 <= 1e-5",
            "errors": protocol.operational_noise_envelope.errors,
        },
        "task_level_impact": {
            "status": protocol.task_level_impact.status,
            "sample_count": protocol.task_level_impact.sample_count,
            "metric_differences_pp": protocol.task_level_impact.metric_differences,
            "p95_localization_error_diff": (
                protocol.task_level_impact.p95_localization_error_diff
            ),
            "nonfinite_samples": protocol.task_level_impact.nonfinite_samples,
            "provenance_summary": [
                {
                    "dataset": p["dataset"],
                    "canonical_category": p["canonical_category"],
                    "total_sample_count": p["total_sample_count"],
                    "split_config_hash": p["split_config_hash"],
                    "accepted_for_b1_task_gate": p["accepted_for_b1_task_gate"],
                }
                for p in protocol.task_level_impact.provenance
            ],
            "errors": protocol.task_level_impact.errors,
        },
    }

    limitations = list(result.limitations)
    if not any("Nonstandard" in item or "nonstandard" in item for item in limitations):
        limitations.append(
            "Nonstandard layer set [2,4,6,8] validated on tiny synthetic CUDA model only; "
            "official VisualAD run uses candidate layers [6,12,18,24]."
        )

    command = generation_command or [
        "python",
        "tools/qualify_b1_cuda_equivalence.py",
        "--checkpoint",
        result.checkpoint_path,
        "--expected-sha256",
        result.checkpoint_sha256,
        "--output-dir",
        "docs/phase_b",
    ]

    manifest: dict[str, Any] = {
        "schema_version": 4,
        "phase": "B1",
        "increment": "B1-05",
        "status": result.status,
        "detail": result.detail,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": result.git_sha,
        "generation_command": command,
        "accepted": {
            "task_level_categories": [
                {"dataset": "mvtec", "category": "bottle"},
                {"dataset": "visa", "category": "candle"},
            ]
        },
        "invalidated_previous_evaluation": {
            "task_level_categories": [
                {
                    "dataset": "mvtec",
                    "category": "sample",
                    "reason": (
                        "flat image/ asset directory without train/test/ground_truth; "
                        "not returned by MVTecAdapter"
                    ),
                    "retained_evidence": {
                        "disposition": "removed_superseded_candidate",
                        "former_path": (
                            "docs/phase_b/b1_cuda_equivalence_manifest.pre_b105_candidate.json"
                        ),
                        "sha256": PRE_B105_CANDIDATE_SHA256,
                        "note": (
                            "Superseded raw candidate with repeated measurement dumps; "
                            "final concise manifest retains decision-critical gates."
                        ),
                    },
                }
            ]
        },
        "execution_profile": {
            "identity": "frozen_deterministic_math",
            "path": str(profile_path.relative_to(REPO_ROOT)),
            "sha256": profile_sha256,
            "required_env": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
            "configuration_sha256": configuration_sha256,
            "input_list_sha256": input_list_sha256,
            "project_wide_b2_constraint": True,
            "requested": requested_frozen_profile_settings(),
            "observed": _observed_attestation(observed),
        },
        "strict_status": {
            "status": strict.status,
            "passed": strict.passed,
            "predicate_name": strict.predicate_name,
            "predicate_inputs": strict.predicate_inputs,
            "mismatch_keys": list(strict.mismatch_keys),
            "layer_coverage": strict.layer_coverage.as_dict(),
        },
        "release_closure": {
            "release_closure_available": bool(
                release_closure.get("release_closure_available", True)
            ),
            "selected_b2_profile": release_closure.get("selected_b2_profile"),
            "ten_process_passed": release_closure.get("ten_process_passed", None),
            "ten_process_cross_path_max": release_closure.get(
                "ten_process_cross_path_max"
            ),
            "backend_matrix_summary": release_closure.get("backend_matrix_summary"),
            "production_default_independently_deterministic": release_closure.get(
                "production_default_independently_deterministic"
            ),
            "requires_project_wide_freeze": release_closure.get(
                "requires_project_wide_freeze", True
            ),
            "raw_evidence": release_closure.get("raw_evidence"),
        },
        "checkpoint": {
            "path": result.checkpoint_path,
            "sha256": result.checkpoint_sha256,
            "explicit_cli_required": True,
            "hidden_fallback_paths": False,
        },
        "environment_summary": {
            "python": result.environment.get("python"),
            "pytorch": result.environment.get("pytorch"),
            "cuda_version": result.environment.get("cuda_version"),
            "cudnn_version": result.environment.get("cudnn_version"),
            "nvidia_driver": result.environment.get("nvidia_driver"),
            "gpu_model": result.environment.get("gpu_model"),
            "observed_execution_settings": observed,
        },
        "production_loader": asdict(result.production_loader)
        if result.production_loader
        else None,
        "cpu_suite": asdict(result.cpu_suite) if result.cpu_suite else None,
        "thresholds": {
            "same_chain_max_abs": B1_ATOL,
            "independent_pass_max_abs": B1_ATOL,
            "operational_cross_excess_p95": B1_ATOL,
            "operational_cross_ratio": 1.25,
            "task_level_image_metrics_pp": B1_TASK_IMAGE_METRIC_PP,
            "task_level_pixel_metrics_pp": B1_TASK_PIXEL_METRIC_PP,
            "task_level_aupro_pp": B1_TASK_AUPRO_PP,
            "task_level_boundary_pp": B1_TASK_BOUNDARY_PP,
        },
        "gates": equivalence_protocol,
        "raw_evidence": {
            "path": str(raw_evidence_path),
            "sha256": raw_evidence_sha256,
            "tracked": False,
            "disposition": "ignored_raw_hash_pinned",
            "schema_version": 1,
        },
        "legacy_invalid_task_level_note": (
            "Prior docs/phase_b manifests that used mvtec/sample are retained as "
            "historical evidence but are invalid for the accepted B1 task-level gate."
        ),
        "test_commands": result.test_commands,
        "limitations": limitations,
        "errors": result.errors,
        "production_behavior_changes": False,
        "layer_coverage": strict.layer_coverage.as_dict(),
    }
    missing = [key for key in REQUIRED_TRACKED_MANIFEST_FIELDS if key not in manifest]
    if missing:
        raise ValueError(f"tracked manifest missing required fields: {missing}")
    return manifest


def write_outputs(
    result: B1QualificationResult,
    output_dir: Path,
    protocol: Any | None,
    *,
    release_closure: dict[str, Any] | None = None,
    artifact_root: Path | None = None,
    allow_missing_release_closure: bool = False,
) -> None:
    from rad.artifacts import atomic_write_json, refuse_existing_run

    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("b1_%Y%m%dT%H%M%SZ")
    if artifact_root is None:
        artifact_root = REPO_ROOT / "artifacts" / "phase_b" / "b1_cuda_equivalence" / run_id
    refuse_existing_run(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=False)

    raw_payload: dict[str, Any] = {
        "run_id": run_id,
        "status": result.status,
        "detail": result.detail,
        "feature_diffs": {k: _tensor_diff_dict(v) for k, v in result.feature_diffs.items()},
        "map_diffs": {k: _tensor_diff_dict(v) for k, v in result.map_diffs.items()},
        "block_counts": [asdict(x) for x in result.block_counts],
        "continuation": [asdict(x) for x in result.continuation],
        "latency_ms_sanity": [asdict(x) for x in result.latency],
        "samples": [asdict(s) for s in result.samples],
        "nonstandard_layers": result.nonstandard_layers,
        "environment": result.environment,
    }
    if protocol is not None:
        raw_payload["deterministic_control"] = {
            "status": protocol.deterministic_control.status,
            "decision": protocol.deterministic_control.decision,
            "settings": protocol.deterministic_control.settings,
            "attention_backend": protocol.deterministic_control.attention_backend,
            "comparisons": protocol.deterministic_control.comparisons,
            "deterministic_error": protocol.deterministic_control.deterministic_error,
        }
        raw_payload["algorithmic_same_chain"] = {
            "status": protocol.algorithmic_same_chain.status,
            "feature_diffs": {
                k: _tensor_diff_dict(v)
                for k, v in protocol.algorithmic_same_chain.feature_diffs.items()
            },
            "map_diffs": {
                k: _tensor_diff_dict(v)
                for k, v in protocol.algorithmic_same_chain.map_diffs.items()
            },
            "block_counts": [asdict(x) for x in protocol.algorithmic_same_chain.block_counts],
            "continuation": [asdict(x) for x in protocol.algorithmic_same_chain.continuation],
            "official_candidate_layers_tested": list(
                protocol.algorithmic_same_chain.official_candidate_layers_tested
            ),
            "synthetic_candidate_layers_tested": list(
                protocol.algorithmic_same_chain.synthetic_candidate_layers_tested
            ),
            "nonstandard_official_run_validated": (
                protocol.algorithmic_same_chain.nonstandard_official_run_validated
            ),
            "errors": protocol.algorithmic_same_chain.errors,
        }
        raw_payload["operational_noise_envelope"] = {
            "status": protocol.operational_noise_envelope.status,
            "per_output": protocol.operational_noise_envelope.per_output,
            "errors": protocol.operational_noise_envelope.errors,
            "self_noise_p95": protocol.operational_noise_envelope.self_noise_p95,
            "cross_path_p95": protocol.operational_noise_envelope.cross_path_p95,
            "cross_excess_p95": protocol.operational_noise_envelope.cross_excess_p95,
            "ratio_pass": protocol.operational_noise_envelope.ratio_pass,
            "excess_pass": protocol.operational_noise_envelope.excess_pass,
        }
        tl = protocol.task_level_impact
        raw_payload["task_level_impact"] = {
            "status": tl.status,
            "sample_count": tl.sample_count,
            "metric_differences_pp": tl.metric_differences,
            "metric_raw": tl.metric_raw,
            "per_category_metric_differences_pp": tl.per_category_metric_differences_pp,
            "localization_error_diff_stats": tl.localization_error_diff_stats,
            "map_rel_l2_stats": tl.map_rel_l2_stats,
            "min_map_pearson": tl.min_map_pearson,
            "min_map_spearman": tl.min_map_spearman,
            "min_top1_patch_overlap": tl.min_top1_patch_overlap,
            "max_image_score_diff": tl.max_image_score_diff,
            "nonfinite_samples": tl.nonfinite_samples,
            "provenance": tl.provenance,
            "superseded_evidence": tl.superseded_evidence,
            "samples": [asdict(x) for x in tl.samples],
            "errors": tl.errors,
        }

    raw_path = artifact_root / "raw_evidence.json"
    atomic_write_json(raw_path, raw_payload)
    raw_sha = sha256_file(raw_path)

    configuration = {
        "backbone": BACKBONE,
        "image_size": IMAGE_SIZE,
        "candidate_layers": result.candidate_layers,
        "seed": 111,
        "batch_size": 1,
        "dtype": "float32",
        "eval_mode": True,
        "amp": False,
        "sigma": 4.0,
        "execution_profile": "frozen_deterministic_math",
    }
    config_sha = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode("utf-8")
    ).hexdigest()
    input_list_sha = hashlib.sha256(
        "\n".join(f"{s.sample_id}\t{s.path or ''}" for s in result.samples).encode("utf-8")
    ).hexdigest()
    profile_path = REPO_ROOT / "configs" / "execution" / "frozen_deterministic_math.json"
    if not profile_path.is_file():
        raise FileNotFoundError(f"missing frozen execution profile: {profile_path}")
    profile_sha = sha256_file(profile_path)

    if release_closure is None:
        default_summary = (
            REPO_ROOT
            / "artifacts"
            / "phase_b"
            / "b1_release_closure"
            / "release_closure_summary.json"
        )
        if default_summary.is_file() and not allow_missing_release_closure:
            release_closure = json.loads(default_summary.read_text(encoding="utf-8"))
            release_closure.setdefault("release_closure_available", True)
        else:
            # Describe absence honestly — never synthesize a passing ten-process claim.
            release_closure = {
                "release_closure_available": False,
                "selected_b2_profile": None,
                "ten_process_passed": None,
                "ten_process_cross_path_max": None,
                "backend_matrix_summary": None,
                "production_default_independently_deterministic": None,
                "requires_project_wide_freeze": True,
                "raw_evidence": None,
            }
    else:
        release_closure = dict(release_closure)
        release_closure.setdefault(
            "release_closure_available",
            release_closure.get("ten_process_passed") is not None,
        )

    if protocol is None:
        raise ValueError("protocol is required to write tracked B1 evidence")

    ten_raw = release_closure.get("ten_process_passed", None)
    ten_process_passed: bool | None
    if ten_raw is None:
        ten_process_passed = None
    else:
        ten_process_passed = bool(ten_raw)
    cross_raw = release_closure.get("ten_process_cross_path_max")
    cross_path_max = float(cross_raw) if cross_raw is not None else None

    # Clear provisional errors from qualification-time finalize (no ten-process yet),
    # then apply the single authoritative strict evaluation with assembled evidence.
    result.errors = [
        err
        for err in result.errors
        if not str(err).startswith("strict predicate failed:")
    ]
    strict = _finalize_status(
        result,
        protocol,
        ten_process_passed=ten_process_passed,
        cross_path_max=cross_path_max,
    )

    try:
        relative_raw = str(raw_path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        relative_raw = str(raw_path)

    manifest = build_tracked_b1_manifest(
        result=result,
        protocol=protocol,
        raw_evidence_path=Path(relative_raw),
        raw_evidence_sha256=raw_sha,
        release_closure=release_closure,
        configuration_sha256=config_sha,
        input_list_sha256=input_list_sha,
        profile_sha256=profile_sha,
        strict_status=strict,
    )

    manifest_path = output_dir / "b1_cuda_equivalence_manifest.json"
    if manifest_path.exists():
        legacy = artifact_root / "legacy_pre_b105_manifest.json"
        legacy.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
    atomic_write_json(manifest_path, manifest)

    lines = [
        "# Phase B1: Staged-Backbone CUDA Numerical Equivalence (B1-05)",
        "",
        f"**Status:** `{manifest['status']}`",
        f"**Detail:** `{manifest['detail']}`",
        f"**Strict status:** `{manifest['strict_status']['status']}`",
        f"**Git SHA:** `{result.git_sha}`",
        f"**Timestamp (UTC):** `{manifest['timestamp_utc']}`",
        "**Execution profile:** `frozen_deterministic_math`",
        f"**Profile SHA-256:** `{manifest['execution_profile']['sha256']}`",
        f"**Raw evidence:** `{manifest['raw_evidence']['path']}` "
        f"(sha256=`{manifest['raw_evidence']['sha256']}`)",
        f"**Ten-process:** `{manifest['release_closure']['ten_process_passed']}`",
        "",
        "## Layer coverage",
        "",
        (
            "- Official candidate layers tested: "
            f"`{manifest['layer_coverage']['official_candidate_layers_tested']}`"
        ),
        (
            "- Synthetic candidate layers tested: "
            f"`{manifest['layer_coverage']['synthetic_candidate_layers_tested']}`"
        ),
        (
            "- Nonstandard official run validated: "
            f"`{manifest['layer_coverage']['nonstandard_official_run_validated']}`"
        ),
        "",
        "## Why earlier ~1e-4 appeared",
        "",
        "Independent CUDA forwards under the default attention backend exhibit a "
        "runtime nondeterminism floor near 1e-4. The frozen deterministic math SDP "
        "profile (CUBLAS_WORKSPACE_CONFIG=:4096:8, deterministic algorithms, TF32 "
        "off, flash/mem-efficient SDP off, math SDP on, MHA fastpath off) removes "
        "that floor for official self-noise and staged self-noise.",
        "",
    ]
    equivalence_protocol = manifest.get("gates")
    if equivalence_protocol:
        dc = equivalence_protocol["deterministic_control"]
        lines.extend(
            [
                "## Deterministic control",
                "",
                f"- Decision: `{dc['decision']}` ({dc['status']})",
                f"- Deterministic error: `{dc['deterministic_error']}`",
                "",
            ]
        )
        for pair, val in dc["max_feature_abs"].items():
            lines.append(
                f"- **{pair}**: max_feature={val:.2e}, "
                f"max_map={dc['max_map_abs'][pair]:.2e}"
            )
        g1 = equivalence_protocol["algorithmic_same_chain"]
        g2 = equivalence_protocol["operational_noise_envelope"]
        tl = equivalence_protocol["task_level_impact"]
        lines.extend(
            [
                "",
                "## Gate summaries",
                "",
                f"- Gate 1 same-chain: `{g1['status']}`",
                (
                    f"- Gate 2 operational envelope: `{g2['status']}` "
                    f"(self_p95={g2['self_noise_p95']:.2e}, "
                    f"cross_p95={g2['cross_path_p95']:.2e}, "
                    f"excess={g2['cross_excess_p95']:.2e})"
                ),
                f"- Task-level: `{tl['status']}` on {tl['sample_count']} samples",
                "",
            ]
        )
        for key, val in tl["metric_differences_pp"].items():
            lines.append(f"- {key}: `{val:.4f}` pp")

    if result.limitations:
        lines.extend(["", "## Limitations", ""])
        for item in result.limitations:
            lines.append(f"- {item}")
    if result.errors:
        lines.extend(["", "## Errors", ""])
        for err in result.errors:
            lines.append(f"- {err}")

    (output_dir / "b1_cuda_equivalence_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase B1 CUDA equivalence qualification")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "docs" / "phase_b",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-cpu-suite", action="store_true")
    args = parser.parse_args()
    # Fail closed if required env is absent for non-dry runs.
    if not args.dry_run and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        print(
            "error: CUBLAS_WORKSPACE_CONFIG=:4096:8 is required for B1 qualification",
            file=sys.stderr,
        )
        return 2
    result, protocol = run_qualification(
        checkpoint=args.checkpoint,
        expected_sha256=args.expected_sha256,
        dry_run=args.dry_run,
        skip_cpu_suite=args.skip_cpu_suite,
    )
    if args.dry_run:
        print("dry-run: refusing to overwrite docs/phase_b evidence artifacts")
        print(f"status={result.status}")
        return 0 if result.status in {"passed", "blocked"} else 1
    write_outputs(result, args.output_dir, protocol)
    print(f"status={result.status}")
    print(f"detail={result.detail}")
    if protocol is not None:
        print(f"deterministic_decision={protocol.deterministic_control.decision}")
        print(f"gate1={protocol.algorithmic_same_chain.status}")
        print(f"gate2={protocol.operational_noise_envelope.status}")
        print(f"task_level={protocol.task_level_impact.status}")
    print(f"manifest={args.output_dir / 'b1_cuda_equivalence_manifest.json'}")
    return 0 if result.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

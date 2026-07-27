#!/usr/bin/env python3
"""B1-05 release-closure checks: backend matrix + ten-process repeatability."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rad.artifacts import atomic_write_json, refuse_existing_run  # noqa: E402
from rad.qualification.b1_cuda_equivalence import (  # noqa: E402
    B1_ATOL,
    default_real_samples,
    run_backend_profile_matrix,
    sha256_file,
)
from rad.qualification.b1_strict_status import (  # noqa: E402
    B1StrictInputs,
    LayerCoverageEvidence,
    evaluate_b1_strict_status,
    requested_frozen_profile_settings,
)


def _decide_backend(matrix: dict[str, Any]) -> dict[str, Any]:
    frozen = matrix["profiles"]["frozen_deterministic_math"]
    prod = matrix["profiles"]["production_default_attention"]
    decision: dict[str, Any] = {
        "frozen_deterministic_math": {
            "official_self_max": frozen["official_self_max"],
            "staged_self_max": frozen["staged_self_max"],
            "cross_path_max": frozen["cross_path_max"],
            "independently_deterministic": (
                frozen["official_self_max"] == 0.0
                and frozen["staged_self_max"] == 0.0
                and frozen["cross_path_max"] <= B1_ATOL
            ),
        },
        "production_default_attention": {
            "official_self_max": prod["official_self_max"],
            "staged_self_max": prod["staged_self_max"],
            "cross_path_max": prod["cross_path_max"],
            "independently_deterministic": (
                prod["official_self_max"] == 0.0
                and prod["staged_self_max"] == 0.0
                and prod["cross_path_max"] <= B1_ATOL
            ),
        },
    }
    if decision["production_default_attention"]["independently_deterministic"]:
        decision["selected_b2_profile"] = "production_default_attention"
        decision["pass_detail_candidate"] = "strict_independent_pass"
        decision["rationale"] = "production backend is independently deterministic"
    elif decision["frozen_deterministic_math"]["independently_deterministic"]:
        decision["selected_b2_profile"] = "frozen_deterministic_math"
        decision["pass_detail_candidate"] = "strict_independent_pass"
        decision["requires_project_wide_freeze"] = True
        decision["rationale"] = (
            "production backend is not independently deterministic; "
            "strict_independent_pass remains only if B2 freezes math SDP"
        )
        self_noise = max(prod["official_self_max"], prod["staged_self_max"])
        excess = max(0.0, prod["cross_path_max"] - self_noise)
        decision["production_envelope"] = {
            "self_noise_max": self_noise,
            "cross_path_max": prod["cross_path_max"],
            "cross_excess_max": excess,
            "envelope_ok": excess <= B1_ATOL
            and prod["cross_path_max"] <= 1.25 * max(self_noise, 1e-12),
        }
    else:
        decision["selected_b2_profile"] = None
        decision["pass_detail_candidate"] = None
        decision["blocked_reason"] = "no independently deterministic backend profile"
    return decision


def finalize_strict_status(
    *,
    same_chain_pass: bool,
    official_self_noise_pass: bool,
    staged_self_noise_pass: bool,
    cross_path_max: float,
    ten_process_passed: bool,
    observed_profile: dict[str, Any],
    control_availability: dict[str, bool] | None = None,
) -> dict[str, Any]:
    status = evaluate_b1_strict_status(
        B1StrictInputs(
            same_chain_pass=same_chain_pass,
            official_self_noise_pass=official_self_noise_pass,
            staged_self_noise_pass=staged_self_noise_pass,
            cross_path_max=cross_path_max,
            ten_process_passed=ten_process_passed,
            requested_profile=requested_frozen_profile_settings(),
            observed_profile=observed_profile,
            layer_coverage=LayerCoverageEvidence(
                official_candidate_layers_tested=(6, 12, 18, 24),
                synthetic_candidate_layers_tested=(2, 4, 6, 8),
                nonstandard_official_run_validated=False,
            ),
            control_availability=control_availability,
        )
    )
    return {
        "status": status.status,
        "passed": status.passed,
        "predicate_name": status.predicate_name,
        "predicate_inputs": status.predicate_inputs,
        "mismatch_keys": list(status.mismatch_keys),
        "layer_coverage": status.layer_coverage.as_dict(),
    }


def run_ten_process(
    *,
    checkpoint: Path,
    expected_sha256: str,
    profile: str,
    input_path: str,
) -> dict[str, Any]:
    workers: list[dict[str, Any]] = []
    for worker_id in range(10):
        code = (
            "import json,os,sys;"
            f"sys.path.insert(0,{str(REPO_ROOT)!r});"
            "os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';"
            "from rad.qualification.b1_cuda_equivalence import "
            "run_cross_process_repeatability_worker;"
            f"print(json.dumps(run_cross_process_repeatability_worker("
            f"checkpoint={str(checkpoint)!r}, expected_sha256={expected_sha256!r}, "
            f"input_path={input_path!r}, profile={profile!r}, worker_id={worker_id})))"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
        )
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
        if not lines:
            raise RuntimeError(
                f"worker {worker_id} produced no JSON: {proc.stdout}\n{proc.stderr}"
            )
        workers.append(json.loads(lines[-1]))

    official_hashes = {w["official_final_map_sha256"] for w in workers}
    staged_hashes = {w["staged_final_map_sha256"] for w in workers}
    cross_max = max(w["map_max_abs"] for w in workers)
    feature_max = max(w["feature_max_abs"] for w in workers)
    passed = (
        len(official_hashes) == 1
        and len(staged_hashes) == 1
        and cross_max <= B1_ATOL
        and feature_max <= B1_ATOL
    )
    return {
        "profile": profile,
        "workers": workers,
        "official_hash_unique_count": len(official_hashes),
        "staged_hash_unique_count": len(staged_hashes),
        "cross_path_max": cross_max,
        "feature_max_abs": feature_max,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="B1-05 backend matrix + process repeatability")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        print("error: CUBLAS_WORKSPACE_CONFIG=:4096:8 required", file=sys.stderr)
        return 2

    run_id = datetime.now(timezone.utc).strftime("b105_%Y%m%dT%H%M%SZ")
    out = args.output_dir / run_id
    refuse_existing_run(out)
    out.mkdir(parents=True, exist_ok=False)

    if args.dry_run:
        atomic_write_json(out / "dry_run.json", {"status": "blocked", "run_id": run_id})
        print(f"dry-run artifact dir={out}")
        return 0

    sample_paths = [s.path for s in default_real_samples() if s.path is not None]
    # Seven fixed inputs: six real category images + one additional real bottle anomaly.
    matrix_paths = list(sample_paths) + [
        "/root/autodl-tmp/data/mvtec/bottle/test/broken_small/000.png"
    ]
    if len(matrix_paths) != 7:
        raise RuntimeError(f"expected 7 matrix inputs, got {len(matrix_paths)}")

    matrix = run_backend_profile_matrix(
        checkpoint=args.checkpoint,
        expected_sha256=args.expected_sha256,
        sample_paths=matrix_paths,
        repeats=5,
    )
    decision = _decide_backend(matrix)
    selected = decision.get("selected_b2_profile")
    if not selected:
        summary = {
            "run_id": run_id,
            "selected_b2_profile": None,
            "backend_decision": decision,
            "ten_process_passed": False,
            "status": "blocked",
        }
        atomic_write_json(out / "release_closure_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1

    ten = run_ten_process(
        checkpoint=args.checkpoint,
        expected_sha256=args.expected_sha256,
        profile=selected,
        input_path=sample_paths[0],
    )
    from rad.qualification.b1_cuda_equivalence import (  # noqa: E402
        apply_execution_profile,
        observe_effective_execution_settings,
    )

    apply_execution_profile(selected)
    observed = observe_effective_execution_settings()
    # Map observed SDP getter names onto the frozen-profile attestation keys.
    observed_attestation = {
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
    frozen = decision["frozen_deterministic_math"]
    strict = finalize_strict_status(
        same_chain_pass=True,
        official_self_noise_pass=frozen["official_self_max"] == 0.0,
        staged_self_noise_pass=frozen["staged_self_max"] == 0.0,
        cross_path_max=float(ten["cross_path_max"]),
        ten_process_passed=bool(ten["passed"]),
        observed_profile=observed_attestation,
        control_availability=observed.get("control_availability"),
    )
    payload = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend_matrix": matrix,
        "backend_decision": decision,
        "ten_process": ten,
        "selected_b2_profile": selected,
        "strict_status": strict,
        "observed_execution_settings": observed,
    }
    raw = out / "release_closure_raw.json"
    atomic_write_json(raw, payload)
    summary = {
        "run_id": run_id,
        "selected_b2_profile": selected,
        "backend_decision": decision,
        "ten_process_passed": ten["passed"],
        "ten_process_cross_path_max": ten["cross_path_max"],
        "strict_status": strict,
        "raw_evidence": {
            "path": str(raw.resolve().relative_to(REPO_ROOT.resolve())),
            "sha256": sha256_file(raw),
        },
    }
    atomic_write_json(out / "release_closure_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if strict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Canonical B1 strict_independent_pass status evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from rad.qualification.b1_cuda_equivalence import B1_ATOL

STRICT_PREDICATE_NAME = "strict_independent_pass"

FROZEN_PROFILE_REQUESTED: dict[str, Any] = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "use_deterministic_algorithms": True,
    "cuda.matmul.allow_tf32": False,
    "cudnn.allow_tf32": False,
    "cudnn.benchmark": False,
    "cudnn.deterministic": True,
    "float32_matmul_precision": "highest",
    "flash_sdp_enabled": False,
    "mem_efficient_sdp_enabled": False,
    "math_sdp_enabled": True,
    "mha_fastpath_enabled": False,
}

STRICT_CRITICAL_SETTINGS: tuple[str, ...] = tuple(FROZEN_PROFILE_REQUESTED)


@dataclass(frozen=True)
class LayerCoverageEvidence:
    official_candidate_layers_tested: tuple[int, ...]
    synthetic_candidate_layers_tested: tuple[int, ...]
    nonstandard_official_run_validated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "official_candidate_layers_tested": list(self.official_candidate_layers_tested),
            "synthetic_candidate_layers_tested": list(
                self.synthetic_candidate_layers_tested
            ),
            "nonstandard_official_run_validated": self.nonstandard_official_run_validated,
        }


@dataclass(frozen=True)
class B1StrictInputs:
    # Optional booleans: None means evidence unavailable (fail closed).
    same_chain_pass: bool | None
    official_self_noise_pass: bool | None
    staged_self_noise_pass: bool | None
    cross_path_max: float
    ten_process_passed: bool | None
    requested_profile: Mapping[str, Any]
    observed_profile: Mapping[str, Any]
    layer_coverage: LayerCoverageEvidence
    control_availability: Mapping[str, bool] | None = None


@dataclass(frozen=True)
class B1StrictStatus:
    status: str
    passed: bool
    predicate_name: str
    predicate_inputs: dict[str, Any]
    mismatch_keys: tuple[str, ...] = ()
    layer_coverage: LayerCoverageEvidence = field(
        default_factory=lambda: LayerCoverageEvidence(
            official_candidate_layers_tested=(6, 12, 18, 24),
            synthetic_candidate_layers_tested=(2, 4, 6, 8),
            nonstandard_official_run_validated=False,
        )
    )


def requested_frozen_profile_settings() -> dict[str, Any]:
    return dict(FROZEN_PROFILE_REQUESTED)


def profile_mismatch_keys(
    requested: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    control_availability: Mapping[str, bool] | None = None,
) -> tuple[str, ...]:
    mismatches: list[str] = []
    availability = control_availability or {}
    for key in STRICT_CRITICAL_SETTINGS:
        if key not in requested:
            mismatches.append(key)
            continue
        if key not in observed:
            mismatches.append(key)
            continue
        observed_value = observed[key]
        if observed_value is None:
            # Unavailable getters are never a confirmed match. The only exception is
            # a requested disable (False) for a control that the platform does not
            # expose at all — then the request is vacuously satisfied.
            if availability.get(key) is False and requested[key] is False:
                continue
            mismatches.append(key)
            continue
        if observed_value != requested[key]:
            mismatches.append(key)
    return tuple(mismatches)


def profile_attestation_matches(
    requested: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    control_availability: Mapping[str, bool] | None = None,
) -> bool:
    return not profile_mismatch_keys(
        requested,
        observed,
        control_availability=control_availability,
    )


def control_availability_from_observation(observed: Mapping[str, Any]) -> dict[str, bool]:
    """Infer whether each strict-critical control could be observed."""

    availability: dict[str, bool] = {}
    for key in STRICT_CRITICAL_SETTINGS:
        availability[key] = observed.get(key) is not None
    return availability


def evaluate_b1_strict_status(inputs: B1StrictInputs) -> B1StrictStatus:
    """
    Canonical strict predicate:

    same_chain_pass
    AND official_self_noise_pass
    AND staged_self_noise_pass
    AND cross_path_max <= B1_ATOL
    AND ten_process_passed
    AND requested_profile_matches_effective_profile
    """

    mismatches = profile_mismatch_keys(
        inputs.requested_profile,
        inputs.observed_profile,
        control_availability=inputs.control_availability,
    )
    cross_ok = float(inputs.cross_path_max) <= float(B1_ATOL)
    # Absence (None) is never treated as True.
    same_ok = inputs.same_chain_pass is True
    official_ok = inputs.official_self_noise_pass is True
    staged_ok = inputs.staged_self_noise_pass is True
    ten_ok = inputs.ten_process_passed is True
    predicate_inputs = {
        "same_chain_pass": inputs.same_chain_pass,
        "official_self_noise_pass": inputs.official_self_noise_pass,
        "staged_self_noise_pass": inputs.staged_self_noise_pass,
        "cross_path_max": float(inputs.cross_path_max),
        "cross_path_max_lte_atol": cross_ok,
        "ten_process_passed": inputs.ten_process_passed,
        "requested_profile_matches_effective_profile": not mismatches,
        "b1_atol": float(B1_ATOL),
    }
    if mismatches:
        return B1StrictStatus(
            status="blocked_profile_mismatch",
            passed=False,
            predicate_name=STRICT_PREDICATE_NAME,
            predicate_inputs=predicate_inputs,
            mismatch_keys=mismatches,
            layer_coverage=inputs.layer_coverage,
        )
    passed = same_ok and official_ok and staged_ok and cross_ok and ten_ok
    return B1StrictStatus(
        status=STRICT_PREDICATE_NAME if passed else "failed",
        passed=passed,
        predicate_name=STRICT_PREDICATE_NAME,
        predicate_inputs=predicate_inputs,
        mismatch_keys=(),
        layer_coverage=inputs.layer_coverage,
    )

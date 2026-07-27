"""Canonical B1 strict_independent_pass truth table and profile attestation."""

from __future__ import annotations

from typing import Any

import pytest

from rad.qualification.b1_strict_status import (
    B1StrictInputs,
    LayerCoverageEvidence,
    evaluate_b1_strict_status,
    profile_attestation_matches,
    requested_frozen_profile_settings,
)


def _base_inputs(**overrides: Any) -> B1StrictInputs:
    payload = {
        "same_chain_pass": True,
        "official_self_noise_pass": True,
        "staged_self_noise_pass": True,
        "cross_path_max": 1e-7,
        "ten_process_passed": True,
        "requested_profile": requested_frozen_profile_settings(),
        "observed_profile": requested_frozen_profile_settings(),
        "layer_coverage": LayerCoverageEvidence(
            official_candidate_layers_tested=(6, 12, 18, 24),
            synthetic_candidate_layers_tested=(2, 4, 6, 8),
            nonstandard_official_run_validated=False,
        ),
    }
    payload.update(overrides)
    return B1StrictInputs(**payload)


@pytest.mark.parametrize(
    ("overrides", "expected_status"),
    (
        ({}, "strict_independent_pass"),
        ({"cross_path_max": 1e-4}, "failed"),
        ({"official_self_noise_pass": False}, "failed"),
        ({"staged_self_noise_pass": False}, "failed"),
        ({"same_chain_pass": False}, "failed"),
        ({"ten_process_passed": False}, "failed"),
        (
            {
                "observed_profile": {
                    **requested_frozen_profile_settings(),
                    "cudnn.deterministic": False,
                }
            },
            "blocked_profile_mismatch",
        ),
    ),
)
def test_b1_strict_truth_table(
    overrides: dict[str, Any], expected_status: str
) -> None:
    result = evaluate_b1_strict_status(_base_inputs(**overrides))
    assert result.status == expected_status
    assert result.predicate_name == "strict_independent_pass"
    if expected_status == "strict_independent_pass":
        assert result.passed is True
        assert result.layer_coverage.nonstandard_official_run_validated is False
    else:
        assert result.passed is False


@pytest.mark.parametrize(
    "setting_key",
    (
        "CUBLAS_WORKSPACE_CONFIG",
        "use_deterministic_algorithms",
        "cuda.matmul.allow_tf32",
        "cudnn.allow_tf32",
        "cudnn.benchmark",
        "cudnn.deterministic",
        "float32_matmul_precision",
        "flash_sdp_enabled",
        "mem_efficient_sdp_enabled",
        "math_sdp_enabled",
        "mha_fastpath_enabled",
    ),
)
def test_observed_profile_mismatch_blocks_strict_status(setting_key: str) -> None:
    requested = requested_frozen_profile_settings()
    observed = dict(requested)
    value = observed[setting_key]
    if value is True:
        observed[setting_key] = False
    elif value is False:
        observed[setting_key] = True
    elif isinstance(value, str):
        observed[setting_key] = f"mutated-{value}"
    else:
        observed[setting_key] = "unavailable"
    assert profile_attestation_matches(requested, observed) is False
    result = evaluate_b1_strict_status(
        _base_inputs(requested_profile=requested, observed_profile=observed)
    )
    assert result.status == "blocked_profile_mismatch"
    assert result.passed is False
    assert setting_key in result.mismatch_keys


def test_unavailable_observation_is_not_a_confirmed_match() -> None:
    requested = requested_frozen_profile_settings()
    observed = dict(requested)
    observed["mha_fastpath_enabled"] = None
    assert profile_attestation_matches(requested, observed) is False
    # When the platform explicitly lacks the control and the request is disable,
    # the disable request is vacuously satisfied.
    assert (
        profile_attestation_matches(
            requested,
            observed,
            control_availability={"mha_fastpath_enabled": False},
        )
        is True
    )
    result = evaluate_b1_strict_status(
        _base_inputs(requested_profile=requested, observed_profile=observed)
    )
    assert result.status == "blocked_profile_mismatch"
    assert "mha_fastpath_enabled" in result.mismatch_keys


def test_nonstandard_layer_evidence_is_explicit_not_boolean_claim() -> None:
    result = evaluate_b1_strict_status(_base_inputs())
    coverage = result.layer_coverage.as_dict()
    assert "nonstandard_layers_validated" not in coverage
    assert coverage["official_candidate_layers_tested"] == [6, 12, 18, 24]
    assert coverage["synthetic_candidate_layers_tested"] == [2, 4, 6, 8]
    assert coverage["nonstandard_official_run_validated"] is False


def test_qualification_and_release_closure_share_status_evaluator() -> None:
    from tools import b1_05_release_closure as release
    from tools import qualify_b1_cuda_equivalence as qualify

    assert qualify.evaluate_b1_strict_status is evaluate_b1_strict_status
    assert release.evaluate_b1_strict_status is evaluate_b1_strict_status

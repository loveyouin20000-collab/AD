"""Story 7: B1 strict callers must fail closed on missing/failed evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rad.qualification.b1_strict_status import (
    B1StrictInputs,
    LayerCoverageEvidence,
    evaluate_b1_strict_status,
    requested_frozen_profile_settings,
)
from tests.rad.test_b1_writer_completeness import (
    _FakeDet,
    _FakeNoise,
    _FakeProtocol,
    _FakeSameChain,
    _FakeTask,
    _sample_result,
)
from tools import b1_05_release_closure as release_closure
from tools.qualify_b1_cuda_equivalence import (
    _finalize_status,
    build_tracked_b1_manifest,
    write_outputs,
)

PROFILE_SHA = "7af8dba39633743da0380fef9710940cded655f68c9efa8f84f5a52aeddb3c8d"
CKPT_SHA = "c" * 64
LAYERS = (6, 12, 18, 24)


def _observed() -> dict[str, Any]:
    return dict(requested_frozen_profile_settings())


def _layer_coverage() -> LayerCoverageEvidence:
    return LayerCoverageEvidence(
        official_candidate_layers_tested=LAYERS,
        synthetic_candidate_layers_tested=(2, 4, 6, 8),
        nonstandard_official_run_validated=False,
    )


def _inputs(**overrides: Any) -> B1StrictInputs:
    payload: dict[str, Any] = {
        "same_chain_pass": True,
        "official_self_noise_pass": True,
        "staged_self_noise_pass": True,
        "cross_path_max": 0.0,
        "ten_process_passed": True,
        "requested_profile": requested_frozen_profile_settings(),
        "observed_profile": _observed(),
        "layer_coverage": _layer_coverage(),
    }
    payload.update(overrides)
    return B1StrictInputs(**payload)


@pytest.mark.parametrize(
    ("ten_process_passed", "expect_strict"),
    (
        (None, False),
        (False, False),
        (True, True),
    ),
)
def test_ten_process_evidence_defaults_fail_closed(
    ten_process_passed: bool | None, expect_strict: bool
) -> None:
    result = evaluate_b1_strict_status(
        _inputs(ten_process_passed=ten_process_passed)
    )
    assert result.predicate_inputs["ten_process_passed"] is ten_process_passed
    assert result.passed is expect_strict
    if expect_strict:
        assert result.status == "strict_independent_pass"
    else:
        assert result.status != "strict_independent_pass"


def test_finalize_status_without_ten_process_is_not_strict() -> None:
    result = _sample_result()
    protocol = _FakeProtocol(
        deterministic_control=_FakeDet(),
        algorithmic_same_chain=_FakeSameChain(),
        operational_noise_envelope=_FakeNoise(),
        task_level_impact=_FakeTask(),
    )
    _finalize_status(result, protocol)  # defaults: ten_process unavailable
    assert result.detail != "strict_independent_pass"
    assert result.status != "passed"


def test_write_outputs_without_release_closure_does_not_synthesize_ten_process(
    tmp_path: Path,
) -> None:
    result = _sample_result()
    protocol = _FakeProtocol(
        deterministic_control=_FakeDet(),
        algorithmic_same_chain=_FakeSameChain(),
        operational_noise_envelope=_FakeNoise(),
        task_level_impact=_FakeTask(),
    )
    out = tmp_path / "docs" / "phase_b"
    out.mkdir(parents=True)
    # Ensure default summary path is absent relative to this isolated artifact root.
    write_outputs(
        result,
        out,
        protocol,
        release_closure=None,
        artifact_root=tmp_path / "artifacts" / "phase_b" / "b1_cuda_equivalence" / "run",
        allow_missing_release_closure=True,
    )
    manifest = json.loads(
        (out / "b1_cuda_equivalence_manifest.json").read_text(encoding="utf-8")
    )
    report = (out / "b1_cuda_equivalence_report.md").read_text(encoding="utf-8")
    assert manifest["release_closure"].get("release_closure_available") is False
    assert manifest["release_closure"]["ten_process_passed"] is None
    assert manifest["strict_status"]["passed"] is False
    assert manifest["strict_status"]["status"] != "strict_independent_pass"
    assert manifest["status"] != "passed"
    assert manifest["detail"] != "strict_independent_pass"
    assert "strict_independent_pass" not in report.split("**Strict status:**")[1].splitlines()[0]
    dumped = json.dumps(manifest)
    assert '"ten_process_passed": true' not in dumped.lower().replace(" ", "")


def test_top_level_status_agrees_with_strict_when_ten_process_fails(
    tmp_path: Path,
) -> None:
    result = _sample_result()
    protocol = _FakeProtocol(
        deterministic_control=_FakeDet(),
        algorithmic_same_chain=_FakeSameChain(),
        operational_noise_envelope=_FakeNoise(),
        task_level_impact=_FakeTask(),
    )
    closure = {
        "release_closure_available": True,
        "selected_b2_profile": "frozen_deterministic_math",
        "ten_process_passed": False,
        "ten_process_cross_path_max": 0.0,
        "requires_project_wide_freeze": True,
        "raw_evidence": {
            "path": "artifacts/phase_b/b1_release_closure/raw.json",
            "sha256": "d" * 64,
            "disposition": "ignored_raw_hash_pinned",
        },
    }
    out = tmp_path / "docs" / "phase_b"
    out.mkdir(parents=True)
    write_outputs(
        result,
        out,
        protocol,
        release_closure=closure,
        artifact_root=tmp_path / "artifacts" / "phase_b" / "b1_cuda_equivalence" / "run",
    )
    manifest = json.loads(
        (out / "b1_cuda_equivalence_manifest.json").read_text(encoding="utf-8")
    )
    report = (out / "b1_cuda_equivalence_report.md").read_text(encoding="utf-8")
    assert manifest["strict_status"]["passed"] is False
    assert manifest["status"] == result.status
    assert manifest["detail"] == result.detail
    assert manifest["status"] != "passed"
    assert manifest["detail"] == manifest["strict_status"]["status"]
    assert f"**Status:** `{manifest['status']}`" in report
    assert f"**Strict status:** `{manifest['strict_status']['status']}`" in report


def _gate1_manifest(
    tmp_path: Path,
    *,
    status: str = "passed",
    checkpoint_sha256: str = CKPT_SHA,
    profile_sha256: str = PROFILE_SHA,
    layers: tuple[int, ...] = LAYERS,
) -> Path:
    path = tmp_path / "gate1_manifest.json"
    payload = {
        "checkpoint": {"sha256": checkpoint_sha256},
        "execution_profile": {
            "sha256": profile_sha256,
            "identity": "frozen_deterministic_math",
        },
        "gates": {
            "algorithmic_same_chain": {
                "status": status,
                "official_candidate_layers_tested": list(layers),
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("gate1_kwargs", "expect_same_chain", "expect_strict"),
    (
        ({"missing": True}, None, False),
        ({"status": "failed"}, False, False),
        ({"status": "passed"}, True, True),
        ({"status": "passed", "checkpoint_sha256": "f" * 64}, False, False),
        ({"status": "passed", "profile_sha256": "a" * 64}, False, False),
        ({"status": "passed", "layers": (6, 12, 18)}, False, False),
    ),
)
def test_release_closure_same_chain_attestation(
    tmp_path: Path,
    gate1_kwargs: dict[str, Any],
    expect_same_chain: bool | None,
    expect_strict: bool,
) -> None:
    if gate1_kwargs.get("missing"):
        attestation = release_closure.attest_same_chain_from_gate1(
            None,
            checkpoint_sha256=CKPT_SHA,
            profile_sha256=PROFILE_SHA,
            candidate_layers=LAYERS,
        )
    else:
        path = _gate1_manifest(
            tmp_path,
            status=str(gate1_kwargs.get("status", "passed")),
            checkpoint_sha256=str(gate1_kwargs.get("checkpoint_sha256", CKPT_SHA)),
            profile_sha256=str(gate1_kwargs.get("profile_sha256", PROFILE_SHA)),
            layers=tuple(gate1_kwargs.get("layers", LAYERS)),  # type: ignore[arg-type]
        )
        attestation = release_closure.attest_same_chain_from_gate1(
            path,
            checkpoint_sha256=CKPT_SHA,
            profile_sha256=PROFILE_SHA,
            candidate_layers=LAYERS,
        )
    assert attestation["same_chain_pass"] is expect_same_chain
    strict = release_closure.finalize_strict_status(
        same_chain_pass=attestation["same_chain_pass"],
        official_self_noise_pass=True,
        staged_self_noise_pass=True,
        cross_path_max=0.0,
        ten_process_passed=True,
        observed_profile=_observed(),
    )
    assert strict["passed"] is expect_strict
    if expect_strict:
        assert strict["status"] == "strict_independent_pass"
    else:
        assert strict["status"] != "strict_independent_pass"


@pytest.mark.parametrize(
    ("overrides", "expect_strict"),
    (
        ({}, True),
        ({"ten_process_passed": None}, False),
        ({"ten_process_passed": False}, False),
        ({"same_chain_pass": None}, False),
        ({"same_chain_pass": False}, False),
        ({"cross_path_max": 1e-4}, False),
        (
            {
                "observed_profile": {
                    **requested_frozen_profile_settings(),
                    "cudnn.deterministic": False,
                }
            },
            False,
        ),
    ),
)
def test_canonical_writers_share_truth_table(
    tmp_path: Path, overrides: dict[str, Any], expect_strict: bool
) -> None:
    canonical = evaluate_b1_strict_status(_inputs(**overrides))
    assert canonical.passed is expect_strict

    release = release_closure.finalize_strict_status(
        same_chain_pass=overrides.get("same_chain_pass", True),
        official_self_noise_pass=overrides.get("official_self_noise_pass", True),
        staged_self_noise_pass=overrides.get("staged_self_noise_pass", True),
        cross_path_max=float(overrides.get("cross_path_max", 0.0)),
        ten_process_passed=overrides.get("ten_process_passed", True),
        observed_profile=overrides.get("observed_profile", _observed()),
    )
    assert release["passed"] is canonical.passed
    assert release["status"] is canonical.status or release["status"] == canonical.status

    result = _sample_result()
    protocol = _FakeProtocol(
        deterministic_control=_FakeDet(),
        algorithmic_same_chain=_FakeSameChain(
            status="passed"
            if overrides.get("same_chain_pass", True) is True
            else "failed"
        ),
        operational_noise_envelope=_FakeNoise(
            cross_path_p95=float(overrides.get("cross_path_max", 0.0))
        ),
        task_level_impact=_FakeTask(),
    )
    if "observed_profile" in overrides:
        # Map attestation keys back onto observe_* naming used by writer helpers.
        obs = dict(result.environment["observed_execution_settings"])
        for key, value in overrides["observed_profile"].items():
            if key == "CUBLAS_WORKSPACE_CONFIG":
                obs["cublas_workspace_config"] = value
            else:
                obs[key] = value
        result.environment["observed_execution_settings"] = obs

    ten = overrides.get("ten_process_passed", True)
    closure = {
        "release_closure_available": ten is not None,
        "selected_b2_profile": "frozen_deterministic_math",
        "ten_process_passed": ten,
        "ten_process_cross_path_max": float(overrides.get("cross_path_max", 0.0)),
        "requires_project_wide_freeze": True,
        "raw_evidence": {
            "path": "artifacts/x.json",
            "sha256": "e" * 64,
            "disposition": "ignored_raw_hash_pinned",
        },
    }
    out = tmp_path / f"docs_{abs(hash(str(overrides)))}" / "phase_b"
    out.mkdir(parents=True)
    write_outputs(
        result,
        out,
        protocol,
        release_closure=closure,
        artifact_root=tmp_path
        / f"artifacts_{abs(hash(str(overrides)))}"
        / "phase_b"
        / "b1"
        / "run",
    )
    manifest = json.loads(
        (out / "b1_cuda_equivalence_manifest.json").read_text(encoding="utf-8")
    )
    report = (out / "b1_cuda_equivalence_report.md").read_text(encoding="utf-8")

    # When same_chain is None, writer still sees protocol gate status; force via
    # finalize path covered above. For True/False, protocol drives same_chain.
    if overrides.get("same_chain_pass") is None:
        # Direct manifest build with explicit None ten/same-chain.
        built = build_tracked_b1_manifest(
            result=result,
            protocol=protocol,
            raw_evidence_path=Path("artifacts/x.json"),
            raw_evidence_sha256="e" * 64,
            release_closure=closure,
            configuration_sha256="a" * 64,
            input_list_sha256="b" * 64,
            profile_sha256=PROFILE_SHA,
            strict_status=canonical,
        )
        assert built["strict_status"]["passed"] is canonical.passed
        assert built["status"] != "passed" or canonical.passed
        return

    assert manifest["strict_status"]["passed"] is canonical.passed
    assert manifest["strict_status"]["status"] == canonical.status
    assert manifest["detail"] == canonical.status or (
        not canonical.passed and manifest["status"] != "passed"
    )
    assert (manifest["status"] == "passed") is canonical.passed
    assert f"**Strict status:** `{manifest['strict_status']['status']}`" in report

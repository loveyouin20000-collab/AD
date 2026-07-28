"""Tracked B1 evidence must be tool-generated and decision-complete."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from rad.qualification.b1_cuda_equivalence import (
    B1QualificationResult,
    B1Sample,
    CpuSuiteResult,
    ProductionLoaderResult,
)
from rad.qualification.b1_strict_status import (
    B1StrictInputs,
    LayerCoverageEvidence,
    evaluate_b1_strict_status,
    requested_frozen_profile_settings,
)
from tools.qualify_b1_cuda_equivalence import (
    REQUIRED_TRACKED_MANIFEST_FIELDS,
    build_tracked_b1_manifest,
    write_outputs,
)


@dataclass
class _FakeDiff:
    max_abs: float = 0.0


@dataclass
class _FakeDet:
    status: str = "strict_independent_pass"
    decision: str = "A"
    deterministic_error: Any = None
    comparisons: dict[str, Any] | None = None
    settings: dict[str, Any] | None = None
    attention_backend: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.comparisons is None:
            self.comparisons = {
                "A1_vs_A2": {"max_feature_abs": 0.0, "max_map_abs": 0.0},
                "S1_vs_S2": {"max_feature_abs": 0.0, "max_map_abs": 0.0},
                "A1_vs_S1": {"max_feature_abs": 0.0, "max_map_abs": 0.0},
                "A2_vs_S2": {"max_feature_abs": 0.0, "max_map_abs": 0.0},
            }
        if self.settings is None:
            self.settings = {}
        if self.attention_backend is None:
            self.attention_backend = {}


@dataclass
class _FakeSameChain:
    status: str = "passed"
    feature_diffs: dict[str, Any] | None = None
    map_diffs: dict[str, Any] | None = None
    block_counts: list[Any] | None = None
    continuation: list[Any] | None = None
    continuation_live_tensor_preserved: bool = True
    official_candidate_layers_tested: tuple[int, ...] = (6, 12, 18, 24)
    synthetic_candidate_layers_tested: tuple[int, ...] = (2, 4, 6, 8)
    nonstandard_official_run_validated: bool = False
    errors: list[str] | None = None

    def __post_init__(self) -> None:
        self.feature_diffs = self.feature_diffs or {}
        self.map_diffs = self.map_diffs or {}
        self.block_counts = self.block_counts or []
        self.continuation = self.continuation or []
        self.errors = self.errors or []


@dataclass
class _FakeNoise:
    status: str = "passed"
    self_noise_p95: float = 0.0
    cross_path_p95: float = 0.0
    cross_excess_p95: float = 0.0
    ratio_pass: bool = True
    excess_pass: bool = True
    errors: list[str] | None = None
    per_output: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.errors = self.errors or []
        self.per_output = self.per_output or {}


@dataclass
class _FakeTask:
    status: str = "passed"
    sample_count: int = 2
    metric_differences: dict[str, float] | None = None
    metric_raw: dict[str, Any] | None = None
    per_category_metric_differences_pp: dict[str, Any] | None = None
    localization_error_diff_stats: dict[str, float] | None = None
    map_rel_l2_stats: dict[str, float] | None = None
    min_map_pearson: float = 1.0
    min_map_spearman: float = 1.0
    min_top1_patch_overlap: float = 1.0
    max_image_score_diff: float = 0.0
    p95_localization_error_diff: float = 0.0
    nonfinite_samples: list[str] | None = None
    samples: list[Any] | None = None
    provenance: list[dict[str, Any]] | None = None
    superseded_evidence: dict[str, Any] | None = None
    errors: list[str] | None = None

    def __post_init__(self) -> None:
        self.metric_differences = self.metric_differences or {
            "image_auroc_pp": 0.0,
            "image_ap_pp": 0.0,
            "image_f1_max_pp": 0.0,
            "pixel_auroc_pp": 0.0,
            "pixel_ap_pp": 0.0,
            "pixel_f1_max_pp": 0.0,
            "pixel_aupro_pp": 0.0,
            "boundary_f_score_pp": 0.0,
        }
        self.metric_raw = self.metric_raw or {}
        self.per_category_metric_differences_pp = (
            self.per_category_metric_differences_pp or {}
        )
        self.localization_error_diff_stats = self.localization_error_diff_stats or {
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
        self.map_rel_l2_stats = self.map_rel_l2_stats or {
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
        self.nonfinite_samples = self.nonfinite_samples or []
        self.samples = self.samples or []
        self.provenance = self.provenance or [
            {
                "dataset": "mvtec",
                "canonical_category": "bottle",
                "total_sample_count": 1,
                "split_config_hash": "a" * 64,
                "accepted_for_b1_task_gate": True,
            },
            {
                "dataset": "visa",
                "canonical_category": "candle",
                "total_sample_count": 1,
                "split_config_hash": "b" * 64,
                "accepted_for_b1_task_gate": True,
            },
        ]
        self.superseded_evidence = self.superseded_evidence or {
            "previous_task_level_categories": ["mvtec/sample"],
            "invalid_reason": "fixture",
        }
        self.errors = self.errors or []


@dataclass
class _FakeProtocol:
    deterministic_control: _FakeDet
    algorithmic_same_chain: _FakeSameChain
    operational_noise_envelope: _FakeNoise
    task_level_impact: _FakeTask
    detail: str = "strict_independent_pass"


def _sample_result() -> B1QualificationResult:
    observed = {
        "cublas_workspace_config": ":4096:8",
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
    return B1QualificationResult(
        status="passed",
        git_sha="deadbeef",
        checkpoint_path="/tmp/ckpt.pth",
        checkpoint_sha256="c" * 64,
        environment={
            "python": "3.10.20",
            "pytorch": "2.0.0+cu118",
            "cuda_version": "11.8",
            "cudnn_version": "8700",
            "nvidia_driver": "595.71.05",
            "gpu_model": "NVIDIA GeForce RTX 4090 D",
            "observed_execution_settings": {
                **observed,
                "control_availability": {
                    "CUBLAS_WORKSPACE_CONFIG": True,
                    "use_deterministic_algorithms": True,
                    "cuda.matmul.allow_tf32": True,
                    "cudnn.allow_tf32": True,
                    "cudnn.benchmark": True,
                    "cudnn.deterministic": True,
                    "float32_matmul_precision": True,
                    "flash_sdp_enabled": True,
                    "mem_efficient_sdp_enabled": True,
                    "math_sdp_enabled": True,
                    "mha_fastpath_enabled": True,
                },
            },
        },
        candidate_layers=[6, 12, 18, 24],
        samples=[B1Sample("synthetic", "synthetic", None)],
        production_loader=ProductionLoaderResult(True, "loader", None),
        cpu_suite=CpuSuiteResult(1, 0, 0, True, "1 passed"),
        nonstandard_layers=LayerCoverageEvidence(
            official_candidate_layers_tested=(6, 12, 18, 24),
            synthetic_candidate_layers_tested=(2, 4, 6, 8),
            nonstandard_official_run_validated=False,
        ).as_dict(),
        limitations=["Synthetic completeness fixture"],
        detail="strict_independent_pass",
    )


def test_tracked_manifest_writer_emits_required_decision_fields(tmp_path: Path) -> None:
    result = _sample_result()
    protocol = _FakeProtocol(
        deterministic_control=_FakeDet(),
        algorithmic_same_chain=_FakeSameChain(),
        operational_noise_envelope=_FakeNoise(),
        task_level_impact=_FakeTask(),
    )
    closure = {
        "selected_b2_profile": "frozen_deterministic_math",
        "ten_process_passed": True,
        "ten_process_cross_path_max": 0.0,
        "backend_matrix_summary": "artifacts/phase_b/b1_release_closure/summary.json",
        "production_default_independently_deterministic": False,
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
    for field in REQUIRED_TRACKED_MANIFEST_FIELDS:
        assert field in manifest, f"missing tracked field: {field}"
    assert manifest["accepted"]["task_level_categories"]
    assert manifest["invalidated_previous_evaluation"]["task_level_categories"]
    evidence = manifest["invalidated_previous_evaluation"]["task_level_categories"][0][
        "retained_evidence"
    ]
    assert evidence["sha256"]
    assert evidence["disposition"]
    assert manifest["execution_profile"]["sha256"]
    assert manifest["strict_status"]["status"]
    assert manifest["strict_status"]["predicate_inputs"]
    assert manifest["release_closure"]["ten_process_passed"] is True
    assert manifest["raw_evidence"]["sha256"]
    assert manifest["raw_evidence"]["disposition"] == "ignored_raw_hash_pinned"
    assert "generation_command" in manifest
    assert "nonstandard_layers_validated" not in json.dumps(manifest)
    assert "Synthetic completeness fixture" in report or "limitations" in report.lower()


def test_build_tracked_manifest_rejects_manual_only_decision_claim() -> None:
    result = _sample_result()
    protocol = _FakeProtocol(
        deterministic_control=_FakeDet(),
        algorithmic_same_chain=_FakeSameChain(),
        operational_noise_envelope=_FakeNoise(),
        task_level_impact=_FakeTask(),
    )
    with pytest.raises(ValueError, match="release_closure"):
        build_tracked_b1_manifest(
            result=result,
            protocol=protocol,
            raw_evidence_path=Path("artifacts/x.json"),
            raw_evidence_sha256="e" * 64,
            release_closure=None,
            configuration_sha256="a" * 64,
            input_list_sha256="b" * 64,
            profile_sha256="c" * 64,
        )


def test_strict_status_in_tracked_manifest_uses_story4_semantics() -> None:
    observed = requested_frozen_profile_settings()
    status = evaluate_b1_strict_status(
        B1StrictInputs(
            same_chain_pass=True,
            official_self_noise_pass=True,
            staged_self_noise_pass=True,
            cross_path_max=0.0,
            ten_process_passed=True,
            requested_profile=requested_frozen_profile_settings(),
            observed_profile=observed,
            layer_coverage=LayerCoverageEvidence(
                official_candidate_layers_tested=(6, 12, 18, 24),
                synthetic_candidate_layers_tested=(2, 4, 6, 8),
                nonstandard_official_run_validated=False,
            ),
        )
    )
    assert status.status == "strict_independent_pass"
    assert status.layer_coverage.nonstandard_official_run_validated is False

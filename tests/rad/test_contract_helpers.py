from __future__ import annotations

from pathlib import Path

import pytest

from rad.evaluation.experiment_matrix import ExperimentMatrix, MatrixRow
from tests.rad.contracts.baseline import (
    assert_completed_manifest_contract,
    assert_metric_provenance_contract,
)
from tests.rad.contracts.experiment_matrix import (
    assert_fixed_exit_semantics,
)
from tests.rad.contracts.zero_shot import assert_source_policy_frozen


def test_matrix_helper_rejects_dynamic_fusion_in_fixed_exit_12_equal() -> None:
    def _row(row_id: str, exit_depth: int, fusion: str) -> MatrixRow:
        return MatrixRow(
            id=row_id,
            group="method",
            description="test",
            immutable=True,
            seed=111,
            device="cpu",
            command="python tools/evaluate_adaptive_dataset.py --config x",
            config={"method": {"exit_depth": exit_depth, "fusion": fusion}},
            estimated_gpu_hours=0.1,
            output_dir="out",
        )

    matrix = ExperimentMatrix(
        defaults={},
        rows=(
            _row("fixed_exit_12_equal", 12, "dynamic"),
            _row("fixed_exit_18_equal", 18, "equal"),
            _row("fixed_exit_12_dynamic", 12, "dynamic"),
            _row("fixed_exit_18_dynamic", 18, "dynamic"),
        ),
    )
    with pytest.raises(AssertionError):
        assert_fixed_exit_semantics(matrix)


def test_baseline_helper_rejects_manifest_without_checkpoint_sha() -> None:
    manifest = {
        "status": "completed",
        "schema_version": 1,
        "git_sha": "abc",
        "seed": 111,
        "checkpoint_path": "/tmp/ckpt.pth",
        "config_hashes": {},
        "commands": {"train": [], "test": []},
        "versions": {},
    }
    with pytest.raises(AssertionError, match="checkpoint_sha256"):
        assert_completed_manifest_contract(manifest)


def test_baseline_helper_rejects_metrics_without_aupro_provenance() -> None:
    metrics = {
        "image_auroc": 0.5,
        "image_ap": 0.5,
        "image_f1_max": 0.5,
        "pixel_auroc": 0.5,
        "pixel_ap": 0.5,
        "pixel_f1_max": 0.5,
        "pixel_aupro": 0.5,
    }
    with pytest.raises(AssertionError, match="pixel_aupro_aggregation"):
        assert_metric_provenance_contract(metrics)


def test_zero_shot_helper_rejects_changed_policy_digest(tmp_path: Path) -> None:
    import json

    profiles = {
        "profiles": {
            "balanced": {
                "name": "balanced",
                "gain_threshold": 0.02,
                "kappa": 0.5,
                "map_uncertainty_threshold": 0.5,
                "image_confidence_margin": 0.4,
                "stability_threshold": 1.0,
                "require_map_uncertainty": True,
                "require_image_confidence": True,
                "require_stability": False,
            }
        },
        "schema_version": 1,
    }
    path = tmp_path / "policy_profiles.json"
    path.write_text(json.dumps(profiles), encoding="utf-8")
    from rad.evaluation.zero_shot import load_frozen_policy_profile

    _, digest = load_frozen_policy_profile(path, "balanced")
    profiles["profiles"]["balanced"]["gain_threshold"] = 0.99
    path.write_text(json.dumps(profiles), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        assert_source_policy_frozen(path, "balanced", digest)

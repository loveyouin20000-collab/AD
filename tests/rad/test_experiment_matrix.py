from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from rad.evaluation.experiment_matrix import (
    REQUIRED_ABLATIONS,
    REQUIRED_METHODS,
    ExperimentMatrix,
    estimate_gpu_hours,
    load_experiment_matrix,
    validate_row_immutable,
)
from tests.rad.contracts.experiment_matrix import (
    REQUIRED_FIXED_EXIT_ROWS,
    REQUIRED_SELECTOR_ABLATION_ROWS,
    assert_dataset_backed_evaluation_row,
    assert_fixed_exit_semantics,
    assert_no_superseded_fixed_exit_ids,
    assert_no_superseded_selector_ids,
    assert_selector_ablation_semantics,
)

REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS_YAML = REPO / "configs" / "rad" / "experiments.yaml"


def test_experiments_yaml_exists_and_loads():
    assert EXPERIMENTS_YAML.is_file()
    matrix = load_experiment_matrix(EXPERIMENTS_YAML)
    assert isinstance(matrix, ExperimentMatrix)
    assert len(matrix.rows) >= len(REQUIRED_METHODS) + len(REQUIRED_ABLATIONS)


def test_required_methods_and_ablations_present():
    matrix = load_experiment_matrix(EXPERIMENTS_YAML)
    ids = {r.id for r in matrix.rows}
    missing_m = set(REQUIRED_METHODS) - ids
    missing_a = set(REQUIRED_ABLATIONS) - ids
    assert not missing_m, f"missing methods: {sorted(missing_m)}"
    assert not missing_a, f"missing ablations: {sorted(missing_a)}"


def test_each_row_is_complete_immutable_config():
    matrix = load_experiment_matrix(EXPERIMENTS_YAML)
    for row in matrix.rows:
        validate_row_immutable(row)


def test_row_ids_unique():
    matrix = load_experiment_matrix(EXPERIMENTS_YAML)
    ids = [r.id for r in matrix.rows]
    assert len(ids) == len(set(ids))


def test_estimate_gpu_hours_accounts_for_dual_gpus():
    matrix = load_experiment_matrix(EXPERIMENTS_YAML)
    single = estimate_gpu_hours(matrix, num_gpus=1)
    dual = estimate_gpu_hours(matrix, num_gpus=2)
    assert single["total_gpu_hours"] == pytest.approx(dual["total_gpu_hours"])
    assert dual["num_gpus"] == 2
    assert dual["wall_clock_hours_est"] == pytest.approx(
        single["total_gpu_hours"] / 2.0
    )
    assert dual["wall_clock_hours_est"] < single["wall_clock_hours_est"]


def test_dry_run_cli_prints_commands_and_gpu_hours(tmp_path: Path):
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "run_experiment_matrix.py"),
            "--config",
            str(EXPERIMENTS_YAML),
            "--dry-run",
            "--num-gpus",
            "2",
            "--output-dir",
            str(tmp_path / "matrix"),
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    out = proc.stdout.lower()
    assert "gpu-hours" in out or "gpu_hours" in out
    assert "wall" in out
    assert "command" in out or "python" in out
    assert "2" in proc.stdout
    raw = yaml.safe_load(EXPERIMENTS_YAML.read_text())
    assert raw["defaults"]["zero_shot"]["target_tuning"] is False


def test_joint_ablation_row_uses_joint_entrypoint():
    matrix = load_experiment_matrix(EXPERIMENTS_YAML)
    row = matrix.row_by_id("ablation_training_joint")
    command = row.command
    assert "tools/train_joint.py" in command
    assert "configs/rad/joint.yaml" in command
    assert "--allow-joint" in command
    assert row.estimated_gpu_hours == 2.5
    assert row.raw.get("primary_pipeline") is False
    assert row.raw.get("requires_gates") == ["fusion_stage_passed", "lse_stage_passed"]


def test_fixed_exit_rows_have_fair_semantics():
    matrix = load_experiment_matrix(EXPERIMENTS_YAML)
    ids = {r.id for r in matrix.rows}
    assert_no_superseded_fixed_exit_ids(ids)
    for row_id in REQUIRED_FIXED_EXIT_ROWS:
        assert_fixed_exit_semantics(matrix.row_by_id(row_id))


def test_selector_ablation_rows_have_complete_signal_maps():
    matrix = load_experiment_matrix(EXPERIMENTS_YAML)
    ids = {r.id for r in matrix.rows}
    assert_no_superseded_selector_ids(ids)
    for row_id in REQUIRED_SELECTOR_ABLATION_ROWS:
        assert_selector_ablation_semantics(matrix.row_by_id(row_id))


def test_paper_evaluation_method_rows_use_dataset_cli():
    matrix = load_experiment_matrix(EXPERIMENTS_YAML)
    eval_ids = [
        "static_learned_fusion",
        "dynamic_fusion_only",
        "confidence_only_exit",
        "stability_only_exit",
        "confidence_stability_exit",
        "residual_gain_equal_fusion",
        "full_conservative",
        "full_balanced",
        "full_aggressive",
        "random_exit_matched",
        "oracle_earliest_exit",
        *REQUIRED_FIXED_EXIT_ROWS,
        *REQUIRED_SELECTOR_ABLATION_ROWS,
    ]
    for row_id in eval_ids:
        assert_dataset_backed_evaluation_row(matrix.row_by_id(row_id))


def test_dry_run_effective_configs_expose_signal_maps(tmp_path: Path):
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "run_experiment_matrix.py"),
            "--config",
            str(EXPERIMENTS_YAML),
            "--dry-run",
            "--num-gpus",
            "1",
            "--ids",
            "selector_full,selector_without_stability,fixed_exit_12_equal",
            "--output-dir",
            str(tmp_path / "matrix"),
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "smoke_adaptive_engine" not in proc.stdout
    assert "evaluate_adaptive.py" not in proc.stdout
    assert "evaluate_adaptive_dataset.py" in proc.stdout

    matrix = load_experiment_matrix(EXPERIMENTS_YAML)
    equal = matrix.row_by_id("fixed_exit_12_equal")
    assert equal.config["method"]["fusion"] == "equal"
    stab = matrix.row_by_id("selector_without_stability")
    assert stab.config["selector"]["signals"]["stability"] is False
    assert stab.config["selector"]["signals"]["response"] is True

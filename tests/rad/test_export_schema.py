from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rad.evaluation.paper_tables import (
    PAPER_TABLE_IDS,
    REQUIRED_RELEASE_MANIFEST_KEYS,
    build_release_manifest,
    export_paper_tables,
    load_result_summaries,
)

REPO = Path(__file__).resolve().parents[2]


def test_required_paper_table_ids_are_complete() -> None:
    expected = {
        "main_comparison",
        "fusion_ablation",
        "exit_strategy_comparison",
        "selector_ablation",
        "risk_coverage",
        "oracle_gap",
        "zero_shot_transfer",
    }
    assert set(PAPER_TABLE_IDS) == expected


def test_export_writes_csv_and_latex_for_each_table(tmp_path: Path) -> None:
    results = tmp_path / "results"
    (results / "original_visualad").mkdir(parents=True)
    (results / "original_visualad" / "summary.json").write_text(
        json.dumps(
            {
                "method_id": "original_visualad",
                "pixel_ap": 0.9,
                "pro": 0.88,
                "expected_depth": 24.0,
                "false_safe_exit_rate": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (results / "ablation_dlcm_with_shapley").mkdir(parents=True)
    (results / "ablation_dlcm_with_shapley" / "summary.json").write_text(
        json.dumps(
            {
                "method_id": "ablation_dlcm_with_shapley",
                "pixel_ap": 0.91,
                "pro": 0.89,
                "expected_depth": 24.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (results / "oracle_earliest_exit").mkdir(parents=True)
    (results / "oracle_earliest_exit" / "summary.json").write_text(
        json.dumps(
            {
                "method_id": "oracle_earliest_exit",
                "pixel_ap": 0.95,
                "pro": 0.93,
                "expected_depth": 12.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (results / "zero_shot_transfer").mkdir(parents=True)
    (results / "zero_shot_transfer" / "summary.json").write_text(
        json.dumps(
            {
                "method_id": "zero_shot_transfer",
                "pixel_ap_adaptive": 0.8,
                "pixel_ap_full": 0.85,
                "pro_adaptive": 0.78,
                "pro_full": 0.82,
                "expected_depth": 18.0,
                "false_safe_exit_rate": 0.05,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "paper"
    artifacts = export_paper_tables(results_dir=results, output_dir=out)
    assert set(artifacts.keys()) == set(PAPER_TABLE_IDS)
    for table_id in PAPER_TABLE_IDS:
        assert (out / f"{table_id}.csv").is_file()
        assert (out / f"{table_id}.tex").is_file()
        assert artifacts[table_id]["csv"].endswith(f"{table_id}.csv")
        assert artifacts[table_id]["tex"].endswith(f"{table_id}.tex")


def test_release_manifest_contains_required_provenance(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "full_balanced").mkdir()
    (results / "full_balanced" / "summary.json").write_text(
        json.dumps({"method_id": "full_balanced", "pixel_ap": 0.92, "pro": 0.9})
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "paper"
    export_paper_tables(results_dir=results, output_dir=out)
    manifest_path = out / "release_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = set(REQUIRED_RELEASE_MANIFEST_KEYS) - set(manifest)
    assert not missing, f"missing release keys: {sorted(missing)}"
    assert manifest["git_sha"]
    assert isinstance(manifest["configs"], list)
    assert isinstance(manifest["result_files"], list)
    assert isinstance(manifest["checkpoints"], list)
    assert "environment" in manifest
    assert "hashes" in manifest


def test_load_result_summaries_indexes_by_method_id(tmp_path: Path) -> None:
    root = tmp_path / "results"
    (root / "full_conservative").mkdir(parents=True)
    (root / "full_conservative" / "summary.json").write_text(
        json.dumps({"pixel_ap": 0.1, "pro": 0.2}) + "\n", encoding="utf-8"
    )
    summaries = load_result_summaries(root)
    assert "full_conservative" in summaries
    assert summaries["full_conservative"]["pixel_ap"] == pytest.approx(0.1)


def test_build_release_manifest_is_stable_with_fixture(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "main_comparison.csv").write_text("method_id,pixel_ap\n", encoding="utf-8")
    manifest = build_release_manifest(
        results_dir=results,
        output_dir=paper,
        git_sha="abc123",
        config_paths=[REPO / "configs" / "rad" / "experiments.yaml"],
        checkpoint_paths=[],
        environment={"python": "3.10", "torch": "2.0", "cuda_available": False},
    )
    assert manifest["git_sha"] == "abc123"
    assert any("experiments.yaml" in str(p) for p in manifest["configs"])
    assert any(str(p).endswith("main_comparison.csv") for p in manifest["result_files"])


def test_export_cli_dry_run_and_output_dir(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    out = tmp_path / "paper_out"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "export_paper_tables.py"),
            "--results",
            str(results),
            "--output-dir",
            str(out),
            "--seed",
            "111",
            "--dry-run",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "dry_run=true" in proc.stdout
    assert "table_count=7" in proc.stdout
    assert not out.exists() or not any(out.iterdir())


def test_ci_workflow_runs_cpu_gates() -> None:
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "pytest" in ci
    assert "ruff" in ci.lower() or "ruff check" in ci
    assert "mypy" in ci
    assert "dry-run" in ci or "dry_run" in ci
    assert "autodl" in ci.lower() or "gpu" in ci.lower()


def test_traceability_doc_maps_hypotheses() -> None:
    doc = (REPO / "docs" / "traceability.md").read_text(encoding="utf-8")
    assert "hypothesis" in doc.lower()
    assert "module" in doc.lower()
    assert "test" in doc.lower()
    assert "experiment" in doc.lower()
    assert "paper" in doc.lower()
    for table_id in PAPER_TABLE_IDS:
        assert table_id in doc

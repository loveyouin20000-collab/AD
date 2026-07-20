from __future__ import annotations

import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_static_fixtures_declare_test_fixture_contract() -> None:
    for name in ("policy_profiles.json", "descriptor_stats.json", "split_manifest.json"):
        payload = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
        assert payload["artifact_kind"] == "test_fixture"
        assert payload["eligible_for_evaluation"] is False
        assert payload["schema_version"] == 1


def test_ci_experiments_config_loads() -> None:
    from rad.evaluation.experiment_matrix import load_experiment_matrix

    matrix = load_experiment_matrix(
        Path(__file__).resolve().parents[2] / "configs" / "rad" / "ci" / "experiments.yaml"
    )
    assert len(matrix.rows) == 1
    assert matrix.defaults["data"]["split_manifest"] == "tests/rad/fixtures/split_manifest.json"

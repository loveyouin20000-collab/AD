"""V3 training dry-run tests."""

from __future__ import annotations

from pathlib import Path

from rad.phase_b import b2_dlcm_v3_training as training
from tests.rad.b2_dlcm_v3_fixtures import contract_config


def test_dry_run_flags() -> None:
    summary = training.dry_run_complete_v3_contract_validation(
        config=contract_config(),
        seed=17,
        output_dir="/tmp/v3-dry-run-should-not-exist-xyz",
    )
    assert summary["real_training_started"] is False
    assert summary["development_evaluation_started"] is False
    assert summary["final_content_resolved"] is False
    assert summary["final_materialization_started"] is False
    assert summary["final_evaluation_started"] is False
    assert summary["artifact_written"] is False
    assert summary["run_directory_created"] is False
    assert summary["teacher_forward_count"] == 0
    assert not Path("/tmp/v3-dry-run-should-not-exist-xyz").exists()
    assert summary["category_not_in_model"] is True

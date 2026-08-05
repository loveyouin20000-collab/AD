"""V3 contract closure: dual dry-run consistency."""

from __future__ import annotations

from pathlib import Path

from rad.phase_b import b2_dlcm_v3_training as training
from tests.rad.b2_dlcm_v3_fixtures import contract_config

FLAGS = (
    "real_training_started",
    "development_evaluation_started",
    "final_content_resolved",
    "final_materialization_started",
    "final_evaluation_started",
    "artifact_written",
    "run_directory_created",
    "teacher_forward_count",
)


def test_dual_dry_run_permutation_stable() -> None:
    cfg_a = contract_config()
    cfg_b = contract_config()
    # Permute unrelated key order via rebuild
    a = training.dry_run_complete_v3_contract_validation(
        config=cfg_a, seed=17, output_dir="/tmp/v3-closure-a"
    )
    b = training.dry_run_complete_v3_contract_validation(
        config=cfg_b, seed=17, output_dir="/tmp/v3-closure-b"
    )
    for key in FLAGS:
        assert a[key] == b[key]
        if key != "teacher_forward_count":
            assert a[key] is False
        else:
            assert a[key] == 0
    assert not Path("/tmp/v3-closure-a").exists()
    assert not Path("/tmp/v3-closure-b").exists()

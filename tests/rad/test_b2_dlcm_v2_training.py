"""RED/GREEN tests for B2-05C1 V2 GT-only training selection."""

from __future__ import annotations

from pathlib import Path

from rad.phase_b import b2_dlcm_v2_training as subject
from tests.rad.b2_dlcm_v2_fixtures import contract_config


def test_selector_epoch0_only_replaced_by_primary() -> None:
    sel = subject.GTOnlyCheckpointSelector(min_delta=1e-5)
    assert sel.consider(epoch=0, primary=1.0, secondary=0.5) is True
    # Worse/equal secondary alone cannot replace epoch 0.
    assert sel.consider(epoch=1, primary=1.0, secondary=0.0) is False
    assert sel.consider(epoch=2, primary=0.99999, secondary=0.0) is False
    assert sel.consider(epoch=3, primary=0.9, secondary=9.0) is True
    assert sel.best_epoch == 3


def test_teacher_ignored_by_canonical_and_selector_contract() -> None:
    flags = subject.teacher_must_not_affect_selection()
    assert flags == {
        "teacher_in_primary": False,
        "teacher_in_secondary": False,
        "teacher_in_patience": False,
        "teacher_in_canonical": False,
    }
    # Teacher-worse seed with better GT wins.
    winner = subject.select_canonical_seed_gt_only(
        [
            {"seed": 29, "primary": 0.5, "secondary": 0.4, "best_epoch": 10, "teacher": 9.0},
            {"seed": 17, "primary": 0.6, "secondary": 0.1, "best_epoch": 10, "teacher": 0.01},
            {"seed": 43, "primary": 0.55, "secondary": 0.2, "best_epoch": 10, "teacher": 0.02},
        ]
    )
    assert winner == 29


def test_all_epoch0_fallback_seed_17() -> None:
    assert (
        subject.select_canonical_seed_gt_only(
            [
                {"seed": 43, "primary": 1.0, "secondary": 1.0, "best_epoch": 0},
                {"seed": 29, "primary": 0.1, "secondary": 0.1, "best_epoch": 0},
                {"seed": 17, "primary": 2.0, "secondary": 2.0, "best_epoch": 0},
            ]
        )
        == 17
    )


def test_tie_break_smallest_seed() -> None:
    assert (
        subject.select_canonical_seed_gt_only(
            [
                {"seed": 43, "primary": 0.5, "secondary": 0.2, "best_epoch": 5},
                {"seed": 17, "primary": 0.5, "secondary": 0.2, "best_epoch": 9},
                {"seed": 29, "primary": 0.5, "secondary": 0.2, "best_epoch": 3},
            ]
        )
        == 17
    )


def test_dry_run_flags(tmp_path: Path) -> None:
    summary = subject.dry_run_complete_v2_contract_validation(
        config=contract_config(),
        seed=17,
        output_dir=tmp_path / "out",
    )
    assert summary["real_training_started"] is False
    assert summary["development_evaluation_started"] is False
    assert summary["final_content_resolved"] is False
    assert summary["final_materialization_started"] is False
    assert summary["final_evaluation_started"] is False
    assert summary["artifact_written"] is False
    assert summary["run_directory_created"] is False
    assert summary["teacher_forward_count"] == 0
    assert not (tmp_path / "out").exists()

"""Eligibility and worst-category selection tests."""

from __future__ import annotations

from rad.phase_b import b2_dlcm_v3_training as training


def test_eligibility_macro_and_per_category() -> None:
    ok = training.is_checkpoint_eligible(
        depth24_gt_kl_macro=0.1,
        depth24_uniform_gt_kl_macro=0.2,
        per_category_gt_kl={"bottle": 0.1, "carpet": 0.1},
        per_category_uniform_gt_kl={"bottle": 0.2, "carpet": 0.2},
    )
    assert ok is True
    bad = training.is_checkpoint_eligible(
        depth24_gt_kl_macro=0.1,
        depth24_uniform_gt_kl_macro=0.2,
        per_category_gt_kl={"bottle": 0.1, "carpet": 0.5},
        per_category_uniform_gt_kl={"bottle": 0.2, "carpet": 0.2},
    )
    assert bad is False


def test_ineligible_does_not_replace_or_reset_patience() -> None:
    sel = training.EligibleWorstCategorySelector()
    assert sel.consider(
        epoch=0, worst_category_kl=1.0, macro_kl=1.0, gt_signed=1.0, eligible=False
    )
    assert sel.best_epoch == 0
    assert (
        sel.consider(
            epoch=1, worst_category_kl=0.1, macro_kl=0.1, gt_signed=0.1, eligible=False
        )
        is False
    )
    assert sel.best_epoch == 0
    assert sel.patience_counter == 0


def test_eligible_worst_category_selection_and_ties() -> None:
    sel = training.EligibleWorstCategorySelector()
    sel.consider(epoch=0, worst_category_kl=1.0, macro_kl=1.0, gt_signed=1.0, eligible=False)
    assert sel.consider(
        epoch=2, worst_category_kl=0.4, macro_kl=0.3, gt_signed=0.2, eligible=True
    )
    assert sel.best_epoch == 2
    # worse worst-category rejected
    assert (
        sel.consider(
            epoch=3, worst_category_kl=0.5, macro_kl=0.1, gt_signed=0.0, eligible=True
        )
        is False
    )
    # tie on worst, better macro wins
    assert sel.consider(
        epoch=4, worst_category_kl=0.4, macro_kl=0.2, gt_signed=0.2, eligible=True
    )
    assert sel.best_epoch == 4
    # complete tie keeps earlier
    assert (
        sel.consider(
            epoch=5, worst_category_kl=0.4, macro_kl=0.2, gt_signed=0.2, eligible=True
        )
        is False
    )
    assert sel.best_epoch == 4


def test_no_eligible_keeps_epoch0_and_canonical_fallback() -> None:
    sel = training.EligibleWorstCategorySelector()
    sel.consider(epoch=0, worst_category_kl=1.0, macro_kl=1.0, gt_signed=1.0, eligible=False)
    assert sel.best_epoch == 0
    assert sel.best_eligible is False
    seed = training.select_canonical_seed_category_robust(
        [
            {"seed": 29, "eligible": False, "worst_category_kl": 0.9, "macro_kl": 0.9, "gt_signed": 0.9},
            {"seed": 17, "eligible": False, "worst_category_kl": 1.0, "macro_kl": 1.0, "gt_signed": 1.0},
            {"seed": 43, "eligible": False, "worst_category_kl": 0.8, "macro_kl": 0.8, "gt_signed": 0.8},
        ]
    )
    assert seed == 17


def test_canonical_prefers_eligible_then_worst() -> None:
    seed = training.select_canonical_seed_category_robust(
        [
            {"seed": 17, "eligible": False, "worst_category_kl": 0.01, "macro_kl": 0.01, "gt_signed": 0.01},
            {"seed": 29, "eligible": True, "worst_category_kl": 0.4, "macro_kl": 0.3, "gt_signed": 0.2},
            {"seed": 43, "eligible": True, "worst_category_kl": 0.3, "macro_kl": 0.35, "gt_signed": 0.25},
        ]
    )
    assert seed == 43


def test_teacher_and_development_not_in_selector() -> None:
    flags = training.teacher_must_not_affect_selection()
    assert flags["teacher_in_canonical"] is False
    assert flags["development_in_selector"] is False

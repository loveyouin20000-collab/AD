"""B2-04A TDD: mathematical contracts and artifact contracts for dual targets.

Story 1 covers the frozen mathematics. Story 2 adds the per-sample scientific
record schema, the GT map calibration artifact, the training-only Shapley
normalization artifact, the split coverage / collection / plan identities, and
the pure leakage-access helpers. Neither story persists anything: no CLI, no run
directories, no ``.pt`` writes, no teacher forward, no dataset access.

--------------------------------------------------------------------------------
Contract assumed by the Story 1 tests in this file (mathematics only):

* Spatial maps are logically ``[height, width]``. ``as_spatial_map`` accepts the
  teacher-cache shapes ``[1, 1, H, W]`` / ``[1, H, W]`` as well and always returns
  a detached ``float64`` 2-D tensor. Batched maps are rejected.

* Coalition players are the configuration-driven candidate layers ``<= depth``,
  ascending. Coalitions are enumerated by ascending integer bitmask, local bit
  ``i`` = player position ``i``. Coalition fusion is the equal average, never the
  production sum-preserving fusion.

* GT calibration is fitted on *training* records only, independently per depth,
  over every nonempty coalition, with nearest-rank ceiling quantiles
  ``Q(q) = x[max(1, ceil(q * N)) - 1]`` (zero-based) in ``float64``.

* GT utilities: Pixel AP is delegated to the production
  ``rad.evaluation.paper_metrics._binary_ap`` on the *raw* map; Soft Dice uses a
  linear denominator with ``eps = 1e-6`` on the *calibrated* map; the background
  penalty mixes a Top-1% mean (0.7) and the global background mean (0.3) over
  ``K = max(1, ceil(0.01 * |background|))`` pixels with row-major tie-break on the
  full-image flat index. The abnormal utility is ``0.4 AP + 0.4 Dice - 0.2 P_BG``
  and is never clipped. The normal utility is
  ``1 - (0.7 Top1% + 0.3 Global)`` over ``K = max(1, ceil(0.01 * H * W))``.

* The full-depth teacher reference is reconstructed through the production
  ``rad.models.dlcm.sum_preserving_fusion`` with equal valid weights and must be
  bit-exact against the cached ``full_depth_map``.

* Teacher utility is ``0.5 * (rho + 1) / 2 + 0.5 * Top1%Overlap`` on raw maps
  only, with the documented Spearman degeneracy rules and average-rank Spearman.

* Shapley values are exact enumeration in ``float64`` on empty-centered
  utilities, with efficiency residual ``<= 1e-12``, followed by positive-player
  renormalization or the minimum-harm equal-ties fallback.

Contract assumed by the Story 2 tests in this file (artifacts only):

* One scientific record per sample carries both ``gt_localization`` and
  ``teacher_fidelity`` families under ``depth_targets`` for every configured
  prediction depth, with the complete coalition table (component values only,
  never anomaly maps) and float64 utility / Shapley / allocation numbers.

* Scientific digests come from explicit whitelists over canonical JSON, so
  paths, timestamps, Git state, and file-byte hashes can never enter a
  scientific identity, and unknown fields fail closed.

* GT map calibration and Shapley normalization are source-training-only; the
  normalization axes are ``target_family × prediction_depth × candidate_layer``
  with deterministic two-pass float64 population statistics (ddof=0), and
  standardized values exist only at read time.

* The plan identity binds all seven layered identities plus the ordered record
  hashes, the calibration and normalization artifacts, the frozen contract
  versions, and the upstream teacher/descriptor/split/checkpoint identities.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import rad.phase_b.b2_contribution_targets as subject
import tests.rad.b2_contribution_target_fixtures as fixtures
from rad.evaluation import paper_metrics
from rad.models import dlcm as dlcm_module

CANDIDATE_LAYERS = (6, 12, 18, 24)


def _tensor(values: Any, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype)


def _ramp(shape: tuple[int, ...], *, start: float = 0.0, step: float = 1.0) -> torch.Tensor:
    count = int(np.prod(shape))
    flat = torch.arange(count, dtype=torch.float32) * step + start
    return flat.reshape(shape)


def _error_code(excinfo: pytest.ExceptionInfo[Exception]) -> str:
    return str(getattr(excinfo.value, "code", ""))


# ---------------------------------------------------------------------------
# Constants and contract versions
# ---------------------------------------------------------------------------


def test_contract_constants_are_frozen() -> None:
    assert subject.TAU == 1e-12
    assert subject.SOFT_DICE_EPS == 1e-6
    assert subject.FLOAT64 is torch.float64
    assert subject.DEFAULT_PREDICTION_DEPTHS == (12, 18, 24)
    assert subject.DEFAULT_CANDIDATE_LAYERS == (6, 12, 18, 24)
    assert subject.TOP_PERCENT_FRACTION == 0.01
    assert subject.GT_ABNORMAL_WEIGHTS == (0.4, 0.4, 0.2)
    assert subject.GT_NORMAL_WEIGHTS == (0.7, 0.3)
    assert subject.BACKGROUND_PENALTY_WEIGHTS == (0.7, 0.3)
    assert subject.TEACHER_UTILITY_WEIGHTS == (0.5, 0.5)
    assert subject.GT_CALIBRATION_QUANTILES == (0.01, 0.995)
    for version in (
        subject.COALITION_CONTRACT_VERSION,
        subject.UTILITY_CONTRACT_VERSION,
        subject.SHAPLEY_CONTRACT_VERSION,
        subject.ALLOCATION_CONTRACT_VERSION,
    ):
        assert isinstance(version, str) and version


def test_module_avoids_teacher_loading_and_target_domain() -> None:
    """Story 3 adds run-directory persistence; the other boundaries still hold.

    Persistence is now an explicit Story 3 deliverable, so ``torch.save``,
    ``atomic_write_json``, and ``mkdir`` are expected. Loading a teacher bundle,
    running a backbone, driving a CLI, or reaching the target domain from the
    domain module never becomes acceptable.
    """

    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    for forbidden in (
        "load_teacher_bundle",
        "build_backbone",
        "visa",
        "argparse",
        "torch.backends",
    ):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# Spatial map normalization
# ---------------------------------------------------------------------------


def test_as_spatial_map_accepts_cache_shapes_and_returns_float64() -> None:
    plain = _tensor([[1.0, 2.0], [3.0, 4.0]])
    cache_shaped = plain.reshape(1, 1, 2, 2)
    for candidate in (plain, cache_shaped, plain.reshape(1, 2, 2)):
        result = subject.as_spatial_map(candidate)
        assert result.dtype is torch.float64
        assert tuple(result.shape) == (2, 2)
        assert torch.equal(result, plain.to(torch.float64))


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros(2, 2, 2),
        torch.zeros(3, 1, 2, 2),
        torch.zeros(4),
        torch.zeros((2, 2), dtype=torch.int64),
    ],
)
def test_as_spatial_map_rejects_non_spatial_inputs(bad: torch.Tensor) -> None:
    with pytest.raises(subject.ContributionTargetError):
        subject.as_spatial_map(bad)


def test_as_spatial_map_rejects_nonfinite_values() -> None:
    bad = _tensor([[0.0, float("nan")], [1.0, 2.0]])
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.as_spatial_map(bad)
    assert _error_code(excinfo) == "B2_TARGET_MAP_NONFINITE"


# ---------------------------------------------------------------------------
# Coalition encoding
# ---------------------------------------------------------------------------


def test_players_for_depth_is_configuration_driven() -> None:
    assert subject.players_for_depth(CANDIDATE_LAYERS, 12) == (6, 12)
    assert subject.players_for_depth(CANDIDATE_LAYERS, 18) == (6, 12, 18)
    assert subject.players_for_depth(CANDIDATE_LAYERS, 24) == (6, 12, 18, 24)
    # A different candidate-layer configuration must be honoured verbatim.
    assert subject.players_for_depth((2, 4, 6), 4) == (2, 4)
    assert subject.players_for_depth((2, 4, 6), 6) == (2, 4, 6)
    assert subject.players_for_depth((5,), 9) == (5,)


def test_players_for_depth_rejects_unsorted_or_duplicate_candidate_layers() -> None:
    for bad in ((12, 6, 18, 24), (6, 6, 12), ()):
        with pytest.raises(subject.ContributionTargetError) as excinfo:
            subject.players_for_depth(bad, 24)
        assert _error_code(excinfo) == "B2_TARGET_CANDIDATE_LAYERS_INVALID"


def test_players_for_depth_rejects_depth_without_players() -> None:
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.players_for_depth(CANDIDATE_LAYERS, 5)
    assert _error_code(excinfo) == "B2_TARGET_DEPTH_HAS_NO_PLAYERS"


def test_enumerate_coalitions_orders_by_ascending_bitmask() -> None:
    coalitions = subject.enumerate_coalitions((6, 12))
    assert [item.bitmask for item in coalitions] == [0, 1, 2, 3]
    assert [item.layer_ids for item in coalitions] == [(), (6,), (12,), (6, 12)]


def test_enumerate_coalitions_cardinality_per_depth() -> None:
    expected = {12: 4, 18: 8, 24: 16}
    for depth, count in expected.items():
        players = subject.players_for_depth(CANDIDATE_LAYERS, depth)
        coalitions = subject.enumerate_coalitions(players)
        assert len(coalitions) == count
        assert len([item for item in coalitions if item.bitmask]) == count - 1


def test_enumerate_coalitions_rejects_wrong_player_order() -> None:
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.enumerate_coalitions((12, 6))
    assert _error_code(excinfo) == "B2_TARGET_PLAYER_ORDER_INVALID"


def test_validate_player_order_returns_canonical_players() -> None:
    assert subject.validate_player_order([6, 12, 18]) == (6, 12, 18)
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.validate_player_order([6, 6])
    assert _error_code(excinfo) == "B2_TARGET_PLAYER_ORDER_INVALID"


def test_validate_coalition_order_rejects_permuted_bitmasks() -> None:
    players = (6, 12, 18)
    coalitions = subject.enumerate_coalitions(players)
    subject.validate_coalition_order(coalitions, players)
    permuted = (coalitions[0], coalitions[2], coalitions[1], *coalitions[3:])
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.validate_coalition_order(permuted, players)
    assert _error_code(excinfo) == "B2_TARGET_COALITION_ORDER_INVALID"


def test_validate_coalition_order_rejects_truncated_enumeration() -> None:
    players = (6, 12)
    coalitions = subject.enumerate_coalitions(players)
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.validate_coalition_order(coalitions[:-1], players)
    assert _error_code(excinfo) == "B2_TARGET_COALITION_ORDER_INVALID"


# ---------------------------------------------------------------------------
# Coalition fusion
# ---------------------------------------------------------------------------


def test_fuse_equal_average_is_the_equal_mean_not_a_sum() -> None:
    layer_maps = {6: torch.ones(4, 5), 12: torch.full((4, 5), 3.0)}
    fused = subject.fuse_equal_average(layer_maps, (6, 12))
    assert fused.dtype is torch.float64
    assert tuple(fused.shape) == (4, 5)
    assert torch.equal(fused, torch.full((4, 5), 2.0, dtype=torch.float64))


def test_fuse_equal_average_matches_manual_mean_for_three_players() -> None:
    layer_maps = {
        6: _ramp((4, 5)),
        12: _ramp((4, 5), start=1.0),
        18: _ramp((4, 5), start=5.0, step=2.0),
    }
    fused = subject.fuse_equal_average(layer_maps, (6, 12, 18))
    stacked = torch.stack([layer_maps[layer].to(torch.float64) for layer in (6, 12, 18)])
    assert torch.allclose(fused, stacked.mean(dim=0), atol=0.0, rtol=0.0)


def test_fuse_equal_average_single_player_returns_that_map() -> None:
    layer_maps = {6: _ramp((4, 5)), 12: torch.zeros(4, 5)}
    fused = subject.fuse_equal_average(layer_maps, (12,))
    assert torch.equal(fused, torch.zeros(4, 5, dtype=torch.float64))


def test_fuse_equal_average_empty_coalition_is_zeros_with_template_shape() -> None:
    template = _ramp((4, 5), start=7.0)
    fused = subject.fuse_equal_average({}, (), template=template)
    assert fused.dtype is torch.float64
    assert tuple(fused.shape) == (4, 5)
    assert torch.equal(fused, torch.zeros(4, 5, dtype=torch.float64))


def test_fuse_equal_average_empty_coalition_uses_lowest_layer_as_template() -> None:
    layer_maps = {6: _ramp((4, 5)), 12: _ramp((4, 5), start=2.0)}
    fused = subject.fuse_equal_average(layer_maps, ())
    assert torch.equal(fused, torch.zeros(4, 5, dtype=torch.float64))


def test_fuse_equal_average_empty_coalition_without_any_template_fails() -> None:
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.fuse_equal_average({}, ())
    assert _error_code(excinfo) == "B2_TARGET_FUSION_TEMPLATE_MISSING"


def test_fuse_equal_average_rejects_missing_or_misordered_layers() -> None:
    layer_maps = {6: torch.ones(4, 5), 12: torch.ones(4, 5)}
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.fuse_equal_average(layer_maps, (6, 18))
    assert _error_code(excinfo) == "B2_TARGET_FUSION_LAYER_MISSING"
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.fuse_equal_average(layer_maps, (12, 6))
    assert _error_code(excinfo) == "B2_TARGET_PLAYER_ORDER_INVALID"


def test_fuse_equal_average_rejects_shape_mismatch() -> None:
    layer_maps = {6: torch.ones(4, 5), 12: torch.ones(4, 6)}
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.fuse_equal_average(layer_maps, (6, 12))
    assert _error_code(excinfo) == "B2_TARGET_FUSION_SHAPE_MISMATCH"


# ---------------------------------------------------------------------------
# Nearest-rank quantiles and GT calibration
# ---------------------------------------------------------------------------


def test_nearest_rank_quantile_uses_ceiling_rule() -> None:
    values = torch.arange(1, 11, dtype=torch.float64) * 10.0
    assert subject.nearest_rank_quantile(values, 0.0) == 10.0
    assert subject.nearest_rank_quantile(values, 0.01) == 10.0
    assert subject.nearest_rank_quantile(values, 0.05) == 10.0
    assert subject.nearest_rank_quantile(values, 0.11) == 20.0
    assert subject.nearest_rank_quantile(values, 0.5) == 50.0
    assert subject.nearest_rank_quantile(values, 0.995) == 100.0
    assert subject.nearest_rank_quantile(values, 1.0) == 100.0


def test_nearest_rank_quantile_single_value() -> None:
    values = torch.tensor([2.5], dtype=torch.float64)
    assert subject.nearest_rank_quantile(values, 0.01) == 2.5
    assert subject.nearest_rank_quantile(values, 0.995) == 2.5


def test_nearest_rank_quantile_rejects_invalid_inputs() -> None:
    sorted_values = torch.arange(1, 5, dtype=torch.float64)
    for bad_q in (-0.1, 1.1):
        with pytest.raises(subject.ContributionTargetError) as excinfo:
            subject.nearest_rank_quantile(sorted_values, bad_q)
        assert _error_code(excinfo) == "B2_TARGET_QUANTILE_LEVEL_INVALID"
    for bad_values in (
        torch.tensor([3.0, 1.0, 2.0], dtype=torch.float64),
        torch.tensor([], dtype=torch.float64),
        torch.arange(1, 5, dtype=torch.float32),
        torch.zeros((2, 2), dtype=torch.float64),
    ):
        with pytest.raises(subject.ContributionTargetError) as excinfo:
            subject.nearest_rank_quantile(bad_values, 0.5)
        assert _error_code(excinfo) == "B2_TARGET_QUANTILE_INPUT_INVALID"


def _training_sample(sample_id: str, seed: int, depths: tuple[int, ...]) -> Any:
    generator = torch.Generator().manual_seed(seed)
    maps_by_depth = {
        depth: {
            layer: torch.rand((4, 5), generator=generator)
            for layer in subject.players_for_depth(CANDIDATE_LAYERS, depth)
        }
        for depth in depths
    }
    return subject.GtCalibrationSample(
        stable_sample_id=sample_id,
        membership="training",
        maps_by_depth=maps_by_depth,
    )


def _reference_pooled_values(samples: Any, depth: int) -> torch.Tensor:
    players = subject.players_for_depth(CANDIDATE_LAYERS, depth)
    chunks = []
    for sample in samples:
        for bitmask in range(1, 1 << len(players)):
            layer_ids = tuple(
                layer for index, layer in enumerate(players) if bitmask >> index & 1
            )
            stacked = torch.stack(
                [sample.maps_by_depth[depth][layer].to(torch.float64) for layer in layer_ids]
            )
            chunks.append(stacked.mean(dim=0).reshape(-1))
    return torch.sort(torch.cat(chunks)).values


def test_fit_gt_map_calibration_matches_nearest_rank_on_pooled_coalition_values() -> None:
    depths = (12, 18)
    samples = [_training_sample(f"id{index}", 100 + index, depths) for index in range(2)]
    calibration = subject.fit_gt_map_calibration(
        samples,
        candidate_layers=CANDIDATE_LAYERS,
        prediction_depths=depths,
    )
    assert tuple(calibration.by_depth) == depths
    for depth in depths:
        pooled = _reference_pooled_values(samples, depth)
        entry = calibration.by_depth[depth]
        assert entry.q_low == pytest.approx(
            subject.nearest_rank_quantile(pooled, 0.01), abs=1e-12
        )
        assert entry.q_high == pytest.approx(
            subject.nearest_rank_quantile(pooled, 0.995), abs=1e-12
        )
        assert isinstance(entry.q_low, float) and isinstance(entry.q_high, float)


def test_fit_gt_map_calibration_records_per_depth_coalition_counts() -> None:
    depths = (12, 18, 24)
    samples = [_training_sample(f"id{index}", 200 + index, depths) for index in range(3)]
    calibration = subject.fit_gt_map_calibration(
        samples,
        candidate_layers=CANDIDATE_LAYERS,
        prediction_depths=depths,
    )
    expected_coalitions = {12: 3, 18: 7, 24: 15}
    for depth, count in expected_coalitions.items():
        entry = calibration.by_depth[depth]
        assert entry.players == subject.players_for_depth(CANDIDATE_LAYERS, depth)
        assert entry.nonempty_coalition_count == count
        assert entry.training_sample_count == 3
        assert entry.value_count == 3 * count * 4 * 5
    assert calibration.statistics_dtype == "float64"
    assert calibration.contract_version == subject.UTILITY_CONTRACT_VERSION
    assert calibration.ordered_training_stable_sample_ids == ("id0", "id1", "id2")
    assert calibration.candidate_layers == CANDIDATE_LAYERS
    assert calibration.prediction_depths == depths


def test_fit_gt_map_calibration_is_independent_per_depth() -> None:
    depths = (12, 24)
    samples = [_training_sample("only", 7, depths)]
    calibration = subject.fit_gt_map_calibration(
        samples,
        candidate_layers=CANDIDATE_LAYERS,
        prediction_depths=depths,
    )
    low = {depth: calibration.by_depth[depth].q_low for depth in depths}
    high = {depth: calibration.by_depth[depth].q_high for depth in depths}
    assert (low[12], high[12]) != (low[24], high[24])


@pytest.mark.parametrize("membership", ["calibration", "evaluation"])
def test_fit_gt_map_calibration_rejects_non_training_records(membership: str) -> None:
    depths = (12,)
    samples = [_training_sample("id0", 3, depths)]
    leaking = subject.GtCalibrationSample(
        stable_sample_id="id1",
        membership=membership,
        maps_by_depth=samples[0].maps_by_depth,
    )
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.fit_gt_map_calibration(
            [*samples, leaking],
            candidate_layers=CANDIDATE_LAYERS,
            prediction_depths=depths,
        )
    assert _error_code(excinfo) == "B2_TARGET_CALIBRATION_LEAKAGE"


def test_fit_gt_map_calibration_rejects_duplicate_and_empty_inputs() -> None:
    depths = (12,)
    sample = _training_sample("dup", 11, depths)
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.fit_gt_map_calibration(
            [sample, sample],
            candidate_layers=CANDIDATE_LAYERS,
            prediction_depths=depths,
        )
    assert _error_code(excinfo) == "B2_TARGET_CALIBRATION_DUPLICATE_SAMPLE"
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.fit_gt_map_calibration(
            [],
            candidate_layers=CANDIDATE_LAYERS,
            prediction_depths=depths,
        )
    assert _error_code(excinfo) == "B2_TARGET_CALIBRATION_SAMPLES_MISSING"


def test_fit_gt_map_calibration_rejects_incomplete_depth_layers() -> None:
    depths = (18,)
    sample = _training_sample("id0", 5, depths)
    broken_maps = {18: {6: sample.maps_by_depth[18][6], 12: sample.maps_by_depth[18][12]}}
    broken = subject.GtCalibrationSample(
        stable_sample_id="id0",
        membership="training",
        maps_by_depth=broken_maps,
    )
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.fit_gt_map_calibration(
            [broken],
            candidate_layers=CANDIDATE_LAYERS,
            prediction_depths=depths,
        )
    assert _error_code(excinfo) == "B2_TARGET_CALIBRATION_MAP_LATTICE_INCOMPLETE"


def test_fit_gt_map_calibration_fails_on_degenerate_constant_maps() -> None:
    depths = (12,)
    constant = {
        12: {layer: torch.full((4, 5), 0.25) for layer in subject.players_for_depth(CANDIDATE_LAYERS, 12)}
    }
    sample = subject.GtCalibrationSample(
        stable_sample_id="flat",
        membership="training",
        maps_by_depth=constant,
    )
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.fit_gt_map_calibration(
            [sample],
            candidate_layers=CANDIDATE_LAYERS,
            prediction_depths=depths,
        )
    assert _error_code(excinfo) == "B2_TARGET_CALIBRATION_DEGENERATE"


def test_apply_gt_calibration_clips_to_unit_interval() -> None:
    anomaly_map = _tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    calibrated = subject.apply_gt_calibration(anomaly_map, 1.0, 3.0)
    expected = torch.tensor(
        [[0.0, 0.0, 0.5], [1.0, 1.0, 1.0]],
        dtype=torch.float64,
    )
    assert calibrated.dtype is torch.float64
    assert torch.equal(calibrated, expected)


def test_apply_gt_calibration_fails_when_high_is_not_above_low() -> None:
    anomaly_map = torch.ones(4, 5)
    for low, high in ((1.0, 1.0), (2.0, 1.0)):
        with pytest.raises(subject.ContributionTargetError) as excinfo:
            subject.apply_gt_calibration(anomaly_map, low, high)
        assert _error_code(excinfo) == "B2_TARGET_CALIBRATION_DEGENERATE"


def test_apply_gt_calibration_rejects_nonfinite_bounds() -> None:
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.apply_gt_calibration(torch.ones(4, 5), 0.0, float("inf"))
    assert _error_code(excinfo) == "B2_TARGET_CALIBRATION_DEGENERATE"


# ---------------------------------------------------------------------------
# GT mask contract
# ---------------------------------------------------------------------------


def test_binarize_and_validate_mask_aligns_with_nearest_neighbour() -> None:
    mask = _tensor([[1.0, 0.0], [0.0, 1.0]])
    aligned = subject.binarize_and_validate_mask(mask, is_anomalous=True, map_shape=(4, 4))
    expected = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    assert aligned.dtype is torch.float64
    assert torch.equal(aligned, expected)


def test_binarize_and_validate_mask_downsamples_without_smoothing() -> None:
    # Striped rows: nearest-neighbour keeps the sampled anomaly rows, while any
    # averaging interpolation would collapse them to 0.5 and lose the anomaly.
    mask = _tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    aligned = subject.binarize_and_validate_mask(mask, is_anomalous=True, map_shape=(2, 4))
    expected = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    assert torch.equal(aligned, expected)


def test_binarize_and_validate_mask_thresholds_above_half() -> None:
    mask = _tensor([[0.0, 0.5], [0.500001, 1.0]])
    binary = subject.binarize_and_validate_mask(mask, is_anomalous=True, map_shape=(2, 2))
    assert torch.equal(binary, torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64))
    assert set(binary.unique().tolist()) <= {0.0, 1.0}


def test_binarize_and_validate_mask_accepts_cache_shaped_mask() -> None:
    mask = _tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    binary = subject.binarize_and_validate_mask(mask, is_anomalous=True, map_shape=(2, 2))
    assert tuple(binary.shape) == (2, 2)
    assert float(binary.sum()) == 1.0


def test_binarize_and_validate_mask_accepts_all_zero_normal_mask() -> None:
    binary = subject.binarize_and_validate_mask(
        torch.zeros(2, 2), is_anomalous=False, map_shape=(4, 5)
    )
    assert tuple(binary.shape) == (4, 5)
    assert float(binary.sum()) == 0.0


def test_binarize_and_validate_mask_rejects_nonzero_normal_mask() -> None:
    mask = _tensor([[0.0, 1.0], [0.0, 0.0]])
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.binarize_and_validate_mask(mask, is_anomalous=False, map_shape=(2, 2))
    assert _error_code(excinfo) == "B2_TARGET_MASK_NORMAL_NOT_EMPTY"


def test_binarize_and_validate_mask_rejects_anomalous_mask_without_anomaly() -> None:
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.binarize_and_validate_mask(
            torch.zeros(2, 2), is_anomalous=True, map_shape=(2, 2)
        )
    assert _error_code(excinfo) == "B2_TARGET_MASK_ANOMALY_MISSING"


def test_binarize_and_validate_mask_rejects_anomalous_mask_without_background() -> None:
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.binarize_and_validate_mask(
            torch.ones(2, 2), is_anomalous=True, map_shape=(2, 2)
        )
    assert _error_code(excinfo) == "B2_TARGET_MASK_BACKGROUND_MISSING"


# ---------------------------------------------------------------------------
# Frozen GT utilities
# ---------------------------------------------------------------------------


def test_soft_dice_matches_linear_denominator_formula() -> None:
    """Authoritative Soft Dice places eps in both numerator and denominator."""

    calibrated = _tensor([[0.5, 0.25], [1.0, 0.0]])
    mask = _tensor([[1.0, 0.0], [1.0, 0.0]])
    intersection = 0.5 * 1.0 + 1.0 * 1.0
    eps = subject.SOFT_DICE_EPS
    expected = (2.0 * intersection + eps) / (1.75 + 2.0 + eps)
    value = subject.soft_dice(calibrated, mask)
    assert isinstance(value, float)
    assert value == pytest.approx(expected, abs=1e-15)


def test_soft_dice_perfect_overlap_is_one_with_shared_eps() -> None:
    mask = _tensor([[1.0, 0.0], [1.0, 0.0]])
    value = subject.soft_dice(mask, mask)
    assert value == pytest.approx(1.0, abs=1e-15)


def test_soft_dice_disjoint_prediction_is_eps_over_mass() -> None:
    calibrated = _tensor([[0.0, 1.0], [0.0, 1.0]])
    mask = _tensor([[1.0, 0.0], [1.0, 0.0]])
    eps = subject.SOFT_DICE_EPS
    expected = eps / (2.0 + 2.0 + eps)
    assert subject.soft_dice(calibrated, mask) == pytest.approx(expected, abs=1e-15)


def test_pixel_ap_raw_delegates_to_production_binary_ap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_map = _tensor([[0.9, 0.1], [0.4, 0.2]])
    mask = _tensor([[1.0, 0.0], [1.0, 0.0]])
    seen: dict[str, np.ndarray] = {}

    def _spy(y_true: np.ndarray, y_score: np.ndarray) -> float:
        seen["y_true"] = np.asarray(y_true)
        seen["y_score"] = np.asarray(y_score)
        return 0.5

    monkeypatch.setattr(paper_metrics, "_binary_ap", _spy)
    assert subject.pixel_ap_raw(raw_map, mask) == 0.5
    assert seen["y_true"].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert seen["y_score"].tolist() == pytest.approx([0.9, 0.1, 0.4, 0.2], abs=1e-7)


def test_pixel_ap_raw_matches_production_value_on_raw_scores() -> None:
    raw_map = _tensor([[9.0, 1.0], [4.0, 2.0]])
    mask = _tensor([[1.0, 0.0], [1.0, 0.0]])
    expected = paper_metrics._binary_ap(
        np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64),
        np.array([9.0, 1.0, 4.0, 2.0], dtype=np.float64),
    )
    assert subject.pixel_ap_raw(raw_map, mask) == pytest.approx(expected, abs=1e-15)


def test_pixel_ap_raw_rejects_shape_mismatch() -> None:
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.pixel_ap_raw(torch.ones(4, 5), torch.ones(4, 6))
    assert _error_code(excinfo) == "B2_TARGET_MASK_SHAPE_MISMATCH"


def test_background_penalty_top_k_rule_and_row_major_tie_break() -> None:
    calibrated = torch.zeros(11, 11)
    mask = torch.zeros(11, 11)
    mask.reshape(-1)[0] = 1.0
    calibrated.reshape(-1)[0] = 1.0  # foreground pixel must be ignored
    for index in (5, 7, 100):
        calibrated.reshape(-1)[index] = 0.9
    components = subject.background_penalty(calibrated, mask)
    assert components.background_pixel_count == 120
    assert components.k == 2
    assert components.top1_percent_indices == (5, 7)
    assert components.top1_percent_mean == pytest.approx(0.9, abs=1e-7)
    assert components.global_mean == pytest.approx(2.7 / 120.0, abs=1e-7)
    assert components.penalty == pytest.approx(
        0.7 * components.top1_percent_mean + 0.3 * components.global_mean, abs=1e-15
    )


def test_background_penalty_small_background_uses_k_one() -> None:
    calibrated = _tensor([[0.2, 0.8], [0.4, 0.1]])
    mask = _tensor([[1.0, 0.0], [0.0, 0.0]])
    components = subject.background_penalty(calibrated, mask)
    assert components.background_pixel_count == 3
    assert components.k == 1
    assert components.top1_percent_indices == (1,)
    assert components.top1_percent_mean == pytest.approx(0.8, abs=1e-7)
    assert components.global_mean == pytest.approx((0.8 + 0.4 + 0.1) / 3.0, abs=1e-7)


def test_background_penalty_breaks_large_tie_sets_by_row_major_index() -> None:
    # 400 background pixels all share one value, so only a row-major-stable
    # selection can return the four lowest full-image flat indices.
    calibrated = torch.full((20, 20), 0.25)
    mask = torch.zeros(20, 20)
    mask.reshape(-1)[0] = 1.0
    components = subject.background_penalty(calibrated, mask)
    assert components.background_pixel_count == 399
    assert components.k == 4
    assert components.top1_percent_indices == (1, 2, 3, 4)
    assert components.top1_percent_mean == pytest.approx(0.25, abs=1e-7)
    assert components.global_mean == pytest.approx(0.25, abs=1e-7)


def test_background_penalty_requires_background_pixels() -> None:
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.background_penalty(torch.zeros(2, 2), torch.ones(2, 2))
    assert _error_code(excinfo) == "B2_TARGET_MASK_BACKGROUND_MISSING"


def test_gt_utility_abnormal_applies_frozen_weights_without_clipping() -> None:
    raw_map = _tensor([[1.0, 9.0], [2.0, 4.0]])
    calibrated = _tensor([[0.0, 1.0], [0.0, 1.0]])
    mask = _tensor([[1.0, 0.0], [1.0, 0.0]])
    components = subject.gt_utility_abnormal(
        raw_map=raw_map, calibrated_map=calibrated, mask=mask
    )
    expected_ap = subject.pixel_ap_raw(raw_map, mask)
    expected_dice = subject.soft_dice(calibrated, mask)
    expected_penalty = subject.background_penalty(calibrated, mask).penalty
    assert components.pixel_ap == pytest.approx(expected_ap, abs=1e-15)
    assert components.soft_dice == pytest.approx(expected_dice, abs=1e-15)
    assert components.background_penalty == pytest.approx(expected_penalty, abs=1e-15)
    assert components.utility == pytest.approx(
        0.4 * expected_ap + 0.4 * expected_dice - 0.2 * expected_penalty, abs=1e-15
    )
    # Worst case here is strictly negative: the utility must not be clipped.
    assert components.utility < 0.0


def test_gt_utility_abnormal_without_background_response_has_zero_penalty() -> None:
    raw_map = _tensor([[1.0, 0.0], [1.0, 0.0]])
    calibrated = _tensor([[1.0, 0.0], [1.0, 0.0]])
    mask = _tensor([[1.0, 0.0], [1.0, 0.0]])
    components = subject.gt_utility_abnormal(
        raw_map=raw_map, calibrated_map=calibrated, mask=mask
    )
    assert components.background_penalty == 0.0
    # Shared eps makes a perfect overlap exactly 1, so this reaches the
    # unpenalized maximum 0.4 * 1 + 0.4 * 1 of the unclipped abnormal utility.
    assert components.pixel_ap == pytest.approx(1.0, abs=1e-15)
    assert components.soft_dice == pytest.approx(1.0, abs=1e-15)
    assert components.utility == pytest.approx(0.4 + 0.4 * components.soft_dice, abs=1e-15)
    assert components.utility == pytest.approx(0.8, abs=1e-15)


def test_gt_utility_normal_uses_full_image_top_k_and_frozen_weights() -> None:
    calibrated = torch.zeros(11, 11)
    for index in (3, 4, 5):
        calibrated.reshape(-1)[index] = 0.5
    components = subject.gt_utility_normal(calibrated_map=calibrated)
    assert components.k == 2
    assert components.top1_percent_mean == pytest.approx(0.5, abs=1e-7)
    assert components.global_mean == pytest.approx(1.5 / 121.0, abs=1e-7)
    assert components.utility == pytest.approx(
        1.0 - (0.7 * components.top1_percent_mean + 0.3 * components.global_mean),
        abs=1e-15,
    )


def test_gt_utility_normal_all_zero_map_is_one() -> None:
    components = subject.gt_utility_normal(calibrated_map=torch.zeros(4, 5))
    assert components.k == 1
    assert components.utility == 1.0


def test_gt_utility_normal_signature_takes_no_mask_or_raw_map() -> None:
    parameters = set(inspect.signature(subject.gt_utility_normal).parameters)
    assert parameters == {"calibrated_map"}


# ---------------------------------------------------------------------------
# Teacher reference reconstruction
# ---------------------------------------------------------------------------


def _cache_maps(shape: tuple[int, int] = (4, 5)) -> dict[int, torch.Tensor]:
    return {
        layer: _ramp((1, 1, *shape), start=float(layer)).contiguous()
        for layer in CANDIDATE_LAYERS
    }


def test_reconstruct_full_depth_teacher_calls_production_fusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    maps = _cache_maps()
    calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    original = dlcm_module.sum_preserving_fusion

    def _spy(
        stacked: torch.Tensor, weights: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        calls.append((stacked, weights, valid_mask))
        return original(stacked, weights, valid_mask)

    monkeypatch.setattr(dlcm_module, "sum_preserving_fusion", _spy)
    reconstructed = subject.reconstruct_full_depth_teacher(
        maps, candidate_layers=CANDIDATE_LAYERS
    )
    assert len(calls) == 1
    stacked, weights, valid_mask = calls[0]
    assert tuple(stacked.shape) == (1, 4, 1, 4, 5)
    assert torch.equal(weights, torch.full((1, 4), 0.25))
    assert bool(valid_mask.all()) and valid_mask.dtype is torch.bool
    assert tuple(reconstructed.shape) == (1, 1, 4, 5)
    assert reconstructed.dtype is torch.float32


def test_reconstruct_full_depth_teacher_is_sum_preserving() -> None:
    maps = _cache_maps()
    reconstructed = subject.reconstruct_full_depth_teacher(
        maps, candidate_layers=CANDIDATE_LAYERS
    )
    expected = sum(maps[layer] for layer in CANDIDATE_LAYERS)
    assert torch.equal(reconstructed, expected)


def test_reconstruct_full_depth_teacher_requires_the_exact_layer_set() -> None:
    maps = _cache_maps()
    del maps[18]
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.reconstruct_full_depth_teacher(maps, candidate_layers=CANDIDATE_LAYERS)
    assert _error_code(excinfo) == "B2_TARGET_TEACHER_LAYER_SET_INVALID"

    extra = _cache_maps()
    extra[30] = extra[24].clone()
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.reconstruct_full_depth_teacher(extra, candidate_layers=CANDIDATE_LAYERS)
    assert _error_code(excinfo) == "B2_TARGET_TEACHER_LAYER_SET_INVALID"


def test_reconstruct_full_depth_teacher_rejects_non_cache_shapes() -> None:
    maps = {layer: torch.ones(4, 5) for layer in CANDIDATE_LAYERS}
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.reconstruct_full_depth_teacher(maps, candidate_layers=CANDIDATE_LAYERS)
    assert _error_code(excinfo) == "B2_TARGET_TEACHER_MAP_SHAPE_INVALID"


def test_verify_full_depth_teacher_bitexact_accepts_identical_tensors() -> None:
    maps = _cache_maps()
    reconstructed = subject.reconstruct_full_depth_teacher(
        maps, candidate_layers=CANDIDATE_LAYERS
    )
    cached = reconstructed.clone()
    assert subject.verify_full_depth_teacher_bitexact(reconstructed, cached) is None


def test_verify_full_depth_teacher_bitexact_reports_first_mismatch() -> None:
    maps = _cache_maps()
    reconstructed = subject.reconstruct_full_depth_teacher(
        maps, candidate_layers=CANDIDATE_LAYERS
    )
    cached = reconstructed.clone()
    cached.reshape(-1)[7] += 0.5
    cached.reshape(-1)[12] += 4.0
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.verify_full_depth_teacher_bitexact(reconstructed, cached)
    assert _error_code(excinfo) == "B2_TARGET_TEACHER_NOT_BITEXACT"
    message = str(excinfo.value)
    assert "first_flat_index=7" in message
    assert "row=1" in message
    assert "col=2" in message
    assert "max_abs_diff=4" in message


def test_verify_full_depth_teacher_bitexact_rejects_shape_or_dtype_drift() -> None:
    reconstructed = torch.ones(1, 1, 4, 5)
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.verify_full_depth_teacher_bitexact(reconstructed, torch.ones(1, 1, 4, 6))
    assert _error_code(excinfo) == "B2_TARGET_TEACHER_SHAPE_MISMATCH"
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.verify_full_depth_teacher_bitexact(
            reconstructed, torch.ones(1, 1, 4, 5, dtype=torch.float64)
        )
    assert _error_code(excinfo) == "B2_TARGET_TEACHER_DTYPE_MISMATCH"


# ---------------------------------------------------------------------------
# Teacher utility
# ---------------------------------------------------------------------------


def test_spearman_fidelity_both_constant_is_raw_one() -> None:
    raw, fidelity = subject.spearman_fidelity(torch.full((4, 5), 2.0), torch.zeros(4, 5))
    assert raw == 1.0
    assert fidelity == 1.0


@pytest.mark.parametrize("constant_side", ["anomaly", "teacher"])
def test_spearman_fidelity_exactly_one_constant_is_raw_zero(constant_side: str) -> None:
    varying = _ramp((4, 5))
    constant = torch.full((4, 5), 3.0)
    if constant_side == "anomaly":
        raw, fidelity = subject.spearman_fidelity(constant, varying)
    else:
        raw, fidelity = subject.spearman_fidelity(varying, constant)
    assert raw == 0.0
    assert fidelity == 0.5


def test_spearman_fidelity_monotone_relationship_is_one() -> None:
    anomaly = _ramp((4, 5), start=1.0)
    teacher = anomaly * 3.0 + 7.0
    raw, fidelity = subject.spearman_fidelity(anomaly, teacher)
    assert raw == pytest.approx(1.0, abs=1e-12)
    assert fidelity == pytest.approx(1.0, abs=1e-12)


def test_spearman_fidelity_reversed_relationship_is_minus_one() -> None:
    anomaly = _ramp((4, 5), start=1.0)
    teacher = -anomaly
    raw, fidelity = subject.spearman_fidelity(anomaly, teacher)
    assert raw == pytest.approx(-1.0, abs=1e-12)
    assert fidelity == pytest.approx(0.0, abs=1e-12)


def test_spearman_fidelity_uses_average_ranks_for_ties() -> None:
    anomaly = _tensor([[1.0, 1.0], [2.0, 3.0]])
    teacher = _tensor([[4.0, 5.0], [6.0, 6.0]])
    raw, fidelity = subject.spearman_fidelity(anomaly, teacher)
    assert raw == pytest.approx(8.0 / 9.0, abs=1e-12)
    assert fidelity == pytest.approx((8.0 / 9.0 + 1.0) / 2.0, abs=1e-12)


def test_spearman_fidelity_returns_named_tuple_pair() -> None:
    result = subject.spearman_fidelity(_ramp((2, 2)), _ramp((2, 2)))
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result.raw == result[0]
    assert result.fidelity == result[1]


def test_top1_overlap_identical_maps_is_one() -> None:
    anomaly = _ramp((11, 11))
    assert subject.top1_overlap(anomaly, anomaly.clone()) == 1.0


def test_top1_overlap_is_intersection_over_k_with_row_major_ties() -> None:
    anomaly = torch.zeros(11, 11)
    teacher = torch.zeros(11, 11)
    for index in (3, 4, 50):
        anomaly.reshape(-1)[index] = 1.0
    for index in (4, 9, 20):
        teacher.reshape(-1)[index] = 1.0
    assert subject.top1_overlap(anomaly, teacher) == pytest.approx(0.5, abs=1e-15)


def test_top1_overlap_disjoint_tops_is_zero() -> None:
    anomaly = torch.zeros(11, 11)
    teacher = torch.zeros(11, 11)
    anomaly.reshape(-1)[0] = 1.0
    anomaly.reshape(-1)[1] = 0.9
    teacher.reshape(-1)[100] = 1.0
    teacher.reshape(-1)[101] = 0.9
    assert subject.top1_overlap(anomaly, teacher) == 0.0


def test_top1_overlap_breaks_large_tie_sets_by_row_major_index() -> None:
    # Every anomaly pixel ties, so the selection must fall back to the lowest
    # row-major indices, which is exactly where the teacher response lives.
    anomaly = torch.zeros(20, 20)
    teacher = torch.zeros(20, 20)
    for index in range(4):
        teacher.reshape(-1)[index] = 1.0
    assert subject.top1_overlap(anomaly, teacher) == 1.0


def test_top1_overlap_small_maps_use_k_one() -> None:
    anomaly = _tensor([[0.1, 0.9], [0.2, 0.3]])
    teacher = _tensor([[0.4, 0.8], [0.2, 0.3]])
    assert subject.top1_overlap(anomaly, teacher) == 1.0


def test_teacher_utility_uses_half_half_weights() -> None:
    anomaly = torch.zeros(11, 11)
    teacher = torch.zeros(11, 11)
    for index in (3, 4, 5):
        anomaly.reshape(-1)[index] = float(index)
        teacher.reshape(-1)[index] = float(10 - index)
    components = subject.teacher_utility(anomaly, teacher)
    expected_raw, expected_fidelity = subject.spearman_fidelity(anomaly, teacher)
    expected_overlap = subject.top1_overlap(anomaly, teacher)
    assert components.spearman_raw == pytest.approx(expected_raw, abs=1e-15)
    assert components.spearman_fidelity == pytest.approx(expected_fidelity, abs=1e-15)
    assert components.top1_overlap == pytest.approx(expected_overlap, abs=1e-15)
    assert components.utility == pytest.approx(
        0.5 * expected_fidelity + 0.5 * expected_overlap, abs=1e-15
    )


def test_teacher_utility_identical_maps_is_one() -> None:
    anomaly = _ramp((11, 11), start=1.0)
    components = subject.teacher_utility(anomaly, anomaly.clone())
    assert components.utility == pytest.approx(1.0, abs=1e-12)


def test_teacher_utility_signature_takes_raw_maps_only() -> None:
    parameters = set(inspect.signature(subject.teacher_utility).parameters)
    assert parameters == {"anomaly_map", "teacher_map"}


def test_teacher_utility_rejects_shape_mismatch() -> None:
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.teacher_utility(torch.ones(4, 5), torch.ones(4, 6))
    assert _error_code(excinfo) == "B2_TARGET_TEACHER_SHAPE_MISMATCH"


# ---------------------------------------------------------------------------
# Empty-coalition centering, exact Shapley, allocation
# ---------------------------------------------------------------------------


def test_center_utilities_subtracts_the_empty_coalition() -> None:
    raw = {0: 0.25, 1: 0.5, 2: 0.75, 3: 1.0}
    centered = subject.center_utilities(raw)
    assert centered[0] == 0.0
    assert centered[1] == pytest.approx(0.25, abs=1e-15)
    assert centered[2] == pytest.approx(0.5, abs=1e-15)
    assert centered[3] == pytest.approx(0.75, abs=1e-15)
    assert list(centered) == [0, 1, 2, 3]


def test_center_utilities_rejects_incomplete_or_invalid_domains() -> None:
    for bad in ({1: 0.5, 2: 0.3, 3: 1.0}, {0: 0.0, 1: 0.5, 2: 0.5}, {}):
        with pytest.raises(subject.ContributionTargetError) as excinfo:
            subject.center_utilities(bad)
        assert _error_code(excinfo) == "B2_TARGET_COALITION_DOMAIN_INVALID"


def test_center_utilities_rejects_nonfinite_utilities() -> None:
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.center_utilities({0: 0.0, 1: float("nan"), 2: 0.5, 3: 1.0})
    assert _error_code(excinfo) == "B2_TARGET_UTILITY_NONFINITE"


def test_exact_shapley_matches_hand_computed_two_player_game() -> None:
    players = (6, 12)
    centered = subject.center_utilities({0: 0.0, 1: 0.2, 2: 0.6, 3: 1.0})
    phi = subject.exact_shapley(players, centered)
    assert phi[6] == pytest.approx(0.3, abs=1e-15)
    assert phi[12] == pytest.approx(0.7, abs=1e-15)
    assert subject.efficiency_residual(phi, centered[3]) <= 1e-12


def test_exact_shapley_is_symmetric_and_efficient_for_three_players() -> None:
    players = (6, 12, 18)
    raw = {0: 0.1}
    for bitmask in range(1, 8):
        raw[bitmask] = 0.1 + 0.3 * bin(bitmask).count("1")
    centered = subject.center_utilities(raw)
    phi = subject.exact_shapley(players, centered)
    assert set(phi) == set(players)
    for player in players:
        assert phi[player] == pytest.approx(0.3, abs=1e-15)
    assert subject.efficiency_residual(phi, centered[7]) <= 1e-12


def test_exact_shapley_uses_standard_permutation_weights() -> None:
    # Asymmetric three-player game: uniform marginal weights would give 0.25 for
    # player 6 instead of the exact Shapley value 7/30.
    players = (6, 12, 18)
    raw = {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.5, 4: 0.4, 5: 0.7, 6: 0.9, 7: 1.2}
    centered = subject.center_utilities(raw)
    phi = subject.exact_shapley(players, centered)
    assert phi[6] == pytest.approx(7.0 / 30.0, abs=1e-15)
    assert phi[12] == pytest.approx(23.0 / 60.0, abs=1e-15)
    assert phi[18] == pytest.approx(7.0 / 12.0, abs=1e-15)
    assert subject.require_shapley_efficiency(phi, centered[7]) <= 1e-12


def test_exact_shapley_gives_dummy_player_zero() -> None:
    players = (6, 12)
    # Player 12 never changes any coalition value.
    centered = subject.center_utilities({0: 0.0, 1: 0.4, 2: 0.0, 3: 0.4})
    phi = subject.exact_shapley(players, centered)
    assert phi[12] == pytest.approx(0.0, abs=1e-15)
    assert phi[6] == pytest.approx(0.4, abs=1e-15)


def test_exact_shapley_returns_float_values_in_player_order() -> None:
    players = (6, 12, 18)
    centered = subject.center_utilities({bitmask: float(bitmask) for bitmask in range(8)})
    phi = subject.exact_shapley(players, centered)
    assert list(phi) == list(players)
    assert all(isinstance(value, float) for value in phi.values())


def test_exact_shapley_rejects_domain_and_player_mismatch() -> None:
    centered = subject.center_utilities({0: 0.0, 1: 0.2, 2: 0.6, 3: 1.0})
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.exact_shapley((6, 12, 18), centered)
    assert _error_code(excinfo) == "B2_TARGET_COALITION_DOMAIN_INVALID"


def test_exact_shapley_rejects_nonzero_empty_coalition() -> None:
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.exact_shapley((6, 12), {0: 0.1, 1: 0.3, 2: 0.7, 3: 1.1})
    assert _error_code(excinfo) == "B2_TARGET_CENTERING_NOT_APPLIED"


def test_efficiency_residual_is_absolute_and_enforced() -> None:
    phi = {6: 0.3, 12: 0.7}
    assert subject.efficiency_residual(phi, 1.0) == pytest.approx(0.0, abs=1e-15)
    assert subject.efficiency_residual(phi, 0.9) == pytest.approx(0.1, abs=1e-15)
    assert subject.require_shapley_efficiency(phi, 1.0) <= 1e-12
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.require_shapley_efficiency(phi, 0.9)
    assert _error_code(excinfo) == "B2_TARGET_SHAPLEY_EFFICIENCY_VIOLATION"


def test_positive_allocation_renormalizes_positive_players() -> None:
    allocation = subject.positive_allocation({6: 0.3, 12: -0.1, 18: 0.7})
    assert allocation[6] == pytest.approx(0.3, abs=1e-15)
    assert allocation[12] == 0.0
    assert allocation[18] == pytest.approx(0.7, abs=1e-15)
    assert sum(allocation.values()) == pytest.approx(1.0, abs=1e-12)


def test_positive_allocation_ignores_values_at_or_below_tau() -> None:
    allocation = subject.positive_allocation({6: 1.0, 12: subject.TAU})
    assert allocation[6] == 1.0
    assert allocation[12] == 0.0


def test_positive_allocation_min_harm_equal_ties_when_all_nonpositive() -> None:
    allocation = subject.positive_allocation({6: -0.5, 12: -0.2, 18: -0.2})
    assert allocation[6] == 0.0
    assert allocation[12] == pytest.approx(0.5, abs=1e-15)
    assert allocation[18] == pytest.approx(0.5, abs=1e-15)
    assert sum(allocation.values()) == pytest.approx(1.0, abs=1e-12)


def test_positive_allocation_all_zero_phi_splits_uniformly() -> None:
    allocation = subject.positive_allocation({6: 0.0, 12: 0.0, 18: 0.0, 24: 0.0})
    assert all(value == pytest.approx(0.25, abs=1e-15) for value in allocation.values())


def test_positive_allocation_single_negative_maximum_takes_everything() -> None:
    allocation = subject.positive_allocation({6: -0.5, 12: -0.2, 18: -0.9})
    assert allocation[12] == 1.0
    assert allocation[6] == 0.0
    assert allocation[18] == 0.0


def test_positive_allocation_is_nonnegative_finite_and_sums_to_one() -> None:
    for phi in (
        {6: 0.1, 12: 0.2, 18: 0.3, 24: 0.4},
        {6: -1.0, 12: 2.0, 18: 0.0, 24: -0.5},
        {6: -3.0, 12: -3.0, 18: -3.0, 24: -3.0},
    ):
        allocation = subject.positive_allocation(phi)
        assert set(allocation) == set(phi)
        assert all(math.isfinite(value) and value >= 0.0 for value in allocation.values())
        assert sum(allocation.values()) == pytest.approx(1.0, abs=1e-12)


def test_positive_allocation_rejects_nonfinite_or_empty_phi() -> None:
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.positive_allocation({6: float("inf"), 12: 0.5})
    assert _error_code(excinfo) == "B2_TARGET_SHAPLEY_NONFINITE"
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.positive_allocation({})
    assert _error_code(excinfo) == "B2_TARGET_ALLOCATION_PLAYERS_MISSING"


# ===========================================================================
# Story 2 — record schema, artifact identities, and leakage-access helpers
# ===========================================================================


@pytest.fixture(scope="module")
def target_fixture() -> Any:
    return fixtures.build_contribution_target_fixture()


@pytest.fixture(scope="module")
def calibration_artifact(target_fixture: Any) -> Any:
    return fixtures.fixture_calibration_artifact(target_fixture)


@pytest.fixture(scope="module")
def target_records(target_fixture: Any) -> Any:
    return fixtures.build_fixture_records(target_fixture)


@pytest.fixture(scope="module")
def normalization(target_fixture: Any, target_records: Any) -> Any:
    return fixtures.build_fixture_normalization(target_fixture, target_records)


def _record_for(records: Any, membership: str, *, label: int) -> Any:
    for row in records:
        if row["split_membership"] == membership and int(row["label"]) == label:
            return row
    raise AssertionError(f"no {membership} record with label {label}")


def _walk(value: Any, path: str = "") -> Any:
    yield path, value
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(item, f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}[{index}]")


def _depth_block(record: Any, depth: int) -> Any:
    return record["depth_targets"][str(depth)]


def _coalition_entry(record: Any, depth: int, bitmask: int) -> Any:
    for entry in _depth_block(record, depth)["coalition_table"]:
        if int(entry["bitmask"]) == bitmask:
            return entry
    raise AssertionError(f"no coalition {bitmask} at depth {depth}")


# ---------------------------------------------------------------------------
# Story 2 constants and contract versions
# ---------------------------------------------------------------------------


def test_story2_contract_constants_are_frozen() -> None:
    assert subject.RECORD_SCHEMA_VERSION == 1
    assert subject.TARGET_FAMILIES == ("gt_localization", "teacher_fidelity")
    assert subject.SPLIT_MEMBERSHIPS == ("training", "calibration", "evaluation")
    assert dict(subject.REQUIRED_SPLIT_COUNTS) == {
        "training": 16,
        "calibration": 8,
        "evaluation": 8,
    }
    assert subject.ACCESS_MODES == ("training_only", "calibration_only", "evaluation_only")
    assert subject.STATISTICS_DTYPE == "float64"
    assert subject.STANDARD_DEVIATION_DDOF == 0
    assert subject.QUANTILE_RULE == "nearest_rank_ceiling"
    assert subject.PRODUCTION_ARTIFACT_KIND == "production"
    assert subject.TEST_FIXTURE_ARTIFACT_KIND == "test_fixture"
    for version in (
        subject.RECORD_CONTRACT_VERSION,
        subject.CALIBRATION_CONTRACT_VERSION,
        subject.NORMALIZATION_CONTRACT_VERSION,
        subject.COLLECTION_CONTRACT_VERSION,
        subject.PLAN_CONTRACT_VERSION,
    ):
        assert isinstance(version, str) and version


def test_domain_module_never_loads_teacher_or_target_domain_or_cli() -> None:
    """Story 3 adds persistence, but the domain module stays teacher/CLI free."""

    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    for forbidden in (
        "load_teacher_bundle",
        "visa",
        "argparse",
        "build_backbone",
        "set_grad_enabled",
    ):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# Hermetic fixture
# ---------------------------------------------------------------------------


def test_fixture_is_split_shaped_with_normal_and_anomalous_samples(
    target_fixture: Any,
) -> None:
    assert target_fixture.artifact_kind == "test_fixture"
    assert len(target_fixture.samples) == 32
    counts = {
        membership: len(target_fixture.by_membership(membership))
        for membership in ("training", "calibration", "evaluation")
    }
    assert counts == {"training": 16, "calibration": 8, "evaluation": 8}
    for membership in counts:
        labels = {sample.label for sample in target_fixture.by_membership(membership)}
        assert labels == {0, 1}
    for sample in target_fixture.samples:
        assert tuple(sample.mask.shape) == target_fixture.map_shape
        assert set(sample.mask.unique().tolist()) <= {0.0, 1.0}
        if sample.is_anomalous:
            assert 0.0 < float(sample.mask.sum()) < float(sample.mask.numel())
        else:
            assert float(sample.mask.sum()) == 0.0
        for depth in target_fixture.prediction_depths:
            expected = subject.players_for_depth(target_fixture.candidate_layers, depth)
            assert tuple(sorted(sample.maps_by_depth[depth])) == expected
            for tensor in sample.maps_by_depth[depth].values():
                assert tuple(tensor.shape) == target_fixture.map_shape


def test_fixture_full_depth_reference_is_bitexact_and_deterministic(
    target_fixture: Any,
) -> None:
    rebuilt = fixtures.build_contribution_target_fixture()
    for left, right in zip(target_fixture.samples, rebuilt.samples, strict=True):
        assert left.stable_sample_id == right.stable_sample_id
        assert torch.equal(left.full_depth_map, right.full_depth_map)
        reconstructed = subject.reconstruct_full_depth_teacher(
            left.maps_by_depth[max(target_fixture.prediction_depths)],
            candidate_layers=target_fixture.candidate_layers,
        )
        assert subject.verify_full_depth_teacher_bitexact(reconstructed, left.full_depth_map) is None


def test_fixture_records_are_never_accepted_by_the_production_gate(
    target_records: Any,
    calibration_artifact: Any,
    normalization: Any,
) -> None:
    for payload in (target_records[0], calibration_artifact, normalization):
        assert payload["artifact_kind"] == "test_fixture"
        with pytest.raises(subject.ContributionTargetError) as excinfo:
            subject.require_production_artifact_kind(payload)
        assert _error_code(excinfo) == "B2_TARGET_TEST_FIXTURE_NOT_ACCEPTED"
    assert subject.require_production_artifact_kind({"artifact_kind": "production"}) is None
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.require_production_artifact_kind({"artifact_kind": "something_else"})
    assert _error_code(excinfo) == "B2_TARGET_ARTIFACT_KIND_INVALID"


def test_fixture_models_the_b2_04b_dual_run_boundary() -> None:
    run_a = fixtures.build_contribution_target_fixture(descriptor_variant="A")
    run_b = fixtures.build_contribution_target_fixture(descriptor_variant="B")
    assert run_a.teacher_cache_scientific_sha256 == run_b.teacher_cache_scientific_sha256
    assert run_a.checkpoint_sha256 == run_b.checkpoint_sha256
    assert (
        run_a.descriptor_collection_scientific_sha256
        != run_b.descriptor_collection_scientific_sha256
    )
    records_a = fixtures.build_fixture_records(run_a)
    records_b = fixtures.build_fixture_records(run_b)
    # Same teacher cache means identical mathematics ...
    for left, right in zip(records_a, records_b, strict=True):
        assert left["depth_targets"] == right["depth_targets"]
        # ... but a different descriptor anchor is a different scientific record.
        assert (
            left["contribution_target_record_scientific_sha256"]
            != right["contribution_target_record_scientific_sha256"]
        )
    normalization_a = fixtures.build_fixture_normalization(run_a, records_a)
    normalization_b = fixtures.build_fixture_normalization(run_b, records_b)
    assert (
        normalization_a["shapley_normalization_scientific_sha256"]
        != normalization_b["shapley_normalization_scientific_sha256"]
    )


# ---------------------------------------------------------------------------
# Upstream identity binding
# ---------------------------------------------------------------------------


def test_bind_upstream_identities_returns_every_scientific_hash(
    target_fixture: Any,
) -> None:
    sample = target_fixture.samples[0]
    upstream = fixtures.fixture_upstream(target_fixture, sample)
    assert upstream.source_teacher_record_scientific_sha256 == (
        sample.teacher_record_scientific_sha256
    )
    assert upstream.descriptor_record_scientific_sha256 == (
        sample.descriptor_record["descriptor_record_scientific_sha256"]
    )
    assert upstream.teacher_cache_scientific_sha256 == (
        target_fixture.teacher_cache_scientific_sha256
    )
    assert upstream.descriptor_collection_scientific_sha256 == (
        target_fixture.descriptor_collection_scientific_sha256
    )
    assert upstream.split_scientific_sha256 == target_fixture.split_scientific_sha256
    assert upstream.checkpoint_sha256 == target_fixture.checkpoint_sha256
    assert upstream.execution_profile_sha256 == target_fixture.execution_profile_sha256


def test_bind_upstream_identities_rejects_teacher_descriptor_mismatch(
    target_fixture: Any,
) -> None:
    sample = target_fixture.samples[0]
    other = target_fixture.samples[1]
    descriptor = dict(sample.descriptor_record)
    descriptor["source_teacher_record_scientific_sha256"] = (
        other.teacher_record_scientific_sha256
    )
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.bind_upstream_identities(
            teacher_record=sample.teacher_record,
            teacher_record_scientific_sha256=sample.teacher_record_scientific_sha256,
            teacher_cache_scientific_sha256=target_fixture.teacher_cache_scientific_sha256,
            teacher_cache_sample_coverage_sha256=(
                target_fixture.teacher_cache_sample_coverage_sha256
            ),
            descriptor_record=descriptor,
            descriptor_collection_scientific_sha256=(
                target_fixture.descriptor_collection_scientific_sha256
            ),
            candidate_layers=target_fixture.candidate_layers,
            prediction_depths=target_fixture.prediction_depths,
        )
    assert _error_code(excinfo) == "B2_TARGET_UPSTREAM_TEACHER_MISMATCH"


def test_bind_upstream_identities_rejects_sample_and_split_mismatch(
    target_fixture: Any,
) -> None:
    sample = target_fixture.by_membership("training")[0]
    other = target_fixture.by_membership("evaluation")[0]

    descriptor = dict(sample.descriptor_record)
    descriptor["stable_sample_id"] = other.stable_sample_id
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.bind_upstream_identities(
            teacher_record=sample.teacher_record,
            teacher_record_scientific_sha256=sample.teacher_record_scientific_sha256,
            teacher_cache_scientific_sha256=target_fixture.teacher_cache_scientific_sha256,
            teacher_cache_sample_coverage_sha256=(
                target_fixture.teacher_cache_sample_coverage_sha256
            ),
            descriptor_record=descriptor,
            descriptor_collection_scientific_sha256=(
                target_fixture.descriptor_collection_scientific_sha256
            ),
            candidate_layers=target_fixture.candidate_layers,
            prediction_depths=target_fixture.prediction_depths,
        )
    assert _error_code(excinfo) == "B2_TARGET_UPSTREAM_SAMPLE_MISMATCH"

    descriptor = dict(sample.descriptor_record)
    descriptor["split_membership"] = "evaluation"
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.bind_upstream_identities(
            teacher_record=sample.teacher_record,
            teacher_record_scientific_sha256=sample.teacher_record_scientific_sha256,
            teacher_cache_scientific_sha256=target_fixture.teacher_cache_scientific_sha256,
            teacher_cache_sample_coverage_sha256=(
                target_fixture.teacher_cache_sample_coverage_sha256
            ),
            descriptor_record=descriptor,
            descriptor_collection_scientific_sha256=(
                target_fixture.descriptor_collection_scientific_sha256
            ),
            candidate_layers=target_fixture.candidate_layers,
            prediction_depths=target_fixture.prediction_depths,
        )
    assert _error_code(excinfo) == "B2_TARGET_UPSTREAM_SPLIT_MISMATCH"


@pytest.mark.parametrize(
    "field",
    ["split_scientific_sha256", "checkpoint_sha256", "execution_profile_sha256"],
)
def test_bind_upstream_identities_rejects_upstream_hash_drift(
    target_fixture: Any, field: str
) -> None:
    sample = target_fixture.samples[0]
    descriptor = dict(sample.descriptor_record)
    descriptor[field] = "0" * 64
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.bind_upstream_identities(
            teacher_record=sample.teacher_record,
            teacher_record_scientific_sha256=sample.teacher_record_scientific_sha256,
            teacher_cache_scientific_sha256=target_fixture.teacher_cache_scientific_sha256,
            teacher_cache_sample_coverage_sha256=(
                target_fixture.teacher_cache_sample_coverage_sha256
            ),
            descriptor_record=descriptor,
            descriptor_collection_scientific_sha256=(
                target_fixture.descriptor_collection_scientific_sha256
            ),
            candidate_layers=target_fixture.candidate_layers,
            prediction_depths=target_fixture.prediction_depths,
        )
    assert _error_code(excinfo) == "B2_TARGET_UPSTREAM_IDENTITY_MISMATCH"


def test_bind_upstream_identities_rejects_lattice_drift_and_bad_hashes(
    target_fixture: Any,
) -> None:
    sample = target_fixture.samples[0]
    descriptor = dict(sample.descriptor_record)
    descriptor["prediction_depths"] = [12, 18]
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.bind_upstream_identities(
            teacher_record=sample.teacher_record,
            teacher_record_scientific_sha256=sample.teacher_record_scientific_sha256,
            teacher_cache_scientific_sha256=target_fixture.teacher_cache_scientific_sha256,
            teacher_cache_sample_coverage_sha256=(
                target_fixture.teacher_cache_sample_coverage_sha256
            ),
            descriptor_record=descriptor,
            descriptor_collection_scientific_sha256=(
                target_fixture.descriptor_collection_scientific_sha256
            ),
            candidate_layers=target_fixture.candidate_layers,
            prediction_depths=target_fixture.prediction_depths,
        )
    assert _error_code(excinfo) == "B2_TARGET_UPSTREAM_LATTICE_MISMATCH"

    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.bind_upstream_identities(
            teacher_record=sample.teacher_record,
            teacher_record_scientific_sha256="not-a-sha256",
            teacher_cache_scientific_sha256=target_fixture.teacher_cache_scientific_sha256,
            teacher_cache_sample_coverage_sha256=(
                target_fixture.teacher_cache_sample_coverage_sha256
            ),
            descriptor_record=sample.descriptor_record,
            descriptor_collection_scientific_sha256=(
                target_fixture.descriptor_collection_scientific_sha256
            ),
            candidate_layers=target_fixture.candidate_layers,
            prediction_depths=target_fixture.prediction_depths,
        )
    assert _error_code(excinfo) == "B2_TARGET_UPSTREAM_HASH_INVALID"


# ---------------------------------------------------------------------------
# GT map calibration artifact
# ---------------------------------------------------------------------------


def test_gt_map_calibration_artifact_carries_depth_bounds_and_identities(
    target_fixture: Any,
    calibration_artifact: Any,
) -> None:
    calibration = fixtures.fit_fixture_calibration(target_fixture)
    assert calibration_artifact["calibration_contract_version"] == (
        subject.CALIBRATION_CONTRACT_VERSION
    )
    assert calibration_artifact["statistics_dtype"] == "float64"
    assert calibration_artifact["quantile_rule"] == "nearest_rank_ceiling"
    assert calibration_artifact["q_low_quantile"] == 0.01
    assert calibration_artifact["q_high_quantile"] == 0.995
    assert calibration_artifact["candidate_layers"] == list(target_fixture.candidate_layers)
    assert calibration_artifact["prediction_depths"] == list(target_fixture.prediction_depths)
    assert calibration_artifact["training_sample_count"] == 16
    training_ids = [
        sample.stable_sample_id for sample in target_fixture.by_membership("training")
    ]
    assert calibration_artifact["ordered_training_stable_sample_ids"] == sorted(training_ids)
    assert set(calibration_artifact["source_teacher_record_scientific_sha256_by_id"]) == set(
        training_ids
    )
    expected_bitmasks = {12: [1, 2, 3], 18: list(range(1, 8)), 24: list(range(1, 16))}
    for depth in target_fixture.prediction_depths:
        entry = calibration_artifact["by_depth"][str(depth)]
        assert entry["prediction_depth"] == depth
        assert entry["ordered_player_layers"] == list(
            subject.players_for_depth(target_fixture.candidate_layers, depth)
        )
        assert entry["nonempty_coalition_bitmasks"] == expected_bitmasks[depth]
        assert entry["nonempty_coalition_count"] == len(expected_bitmasks[depth])
        assert entry["training_sample_count"] == 16
        assert entry["value_count"] == calibration.by_depth[depth].value_count
        assert entry["q_low"] == calibration.by_depth[depth].q_low
        assert entry["q_high"] == calibration.by_depth[depth].q_high
        assert entry["q_high"] > entry["q_low"]
        assert isinstance(entry["q_low"], float) and isinstance(entry["q_high"], float)
    for field in (
        "teacher_cache_scientific_sha256",
        "teacher_cache_sample_coverage_sha256",
        "descriptor_collection_scientific_sha256",
        "split_scientific_sha256",
        "checkpoint_sha256",
        "execution_profile_sha256",
        "gt_map_calibration_training_coverage_sha256",
        "gt_map_calibration_scientific_sha256",
    ):
        assert len(calibration_artifact[field]) == 64


def test_gt_map_calibration_artifact_hash_matches_content_and_ignores_non_science(
    calibration_artifact: Any,
) -> None:
    artifact = copy.deepcopy(calibration_artifact)
    claimed = artifact["gt_map_calibration_scientific_sha256"]
    assert subject.gt_map_calibration_scientific_sha256(artifact) == claimed
    assert subject.validate_gt_map_calibration_artifact(artifact) is None
    for key, value in (
        ("absolute_output_path", "/tmp/gt_map_calibration.pt"),
        ("timestamp", "2026-07-29T00:00:00Z"),
        ("calibration_file_sha256", "0" * 64),
        ("git_branch", "phase-b2-contribution-target-contract"),
    ):
        polluted = dict(artifact)
        polluted[key] = value
        assert subject.gt_map_calibration_scientific_sha256(polluted) == claimed
    drifted = copy.deepcopy(artifact)
    drifted["by_depth"]["12"]["q_high"] = float(drifted["by_depth"]["12"]["q_high"]) + 1.0
    assert subject.gt_map_calibration_scientific_sha256(drifted) != claimed
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.validate_gt_map_calibration_artifact(drifted)
    assert _error_code(excinfo) == "B2_TARGET_CALIBRATION_HASH_MISMATCH"


def test_gt_map_calibration_artifact_rejects_unknown_scientific_fields(
    calibration_artifact: Any,
) -> None:
    polluted = dict(calibration_artifact)
    polluted["sneaky_calibration_field"] = 1
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.gt_map_calibration_scientific_sha256(polluted)
    assert _error_code(excinfo) == "B2_TARGET_CALIBRATION_HASH_SCHEMA_INVALID"


@pytest.mark.parametrize("membership", ["calibration", "evaluation"])
def test_gt_map_calibration_rejects_non_training_samples_from_the_fixture(
    target_fixture: Any, membership: str
) -> None:
    leaking = target_fixture.by_membership(membership)[0]
    samples = [
        *fixtures.fixture_calibration_samples(target_fixture),
        subject.GtCalibrationSample(
            stable_sample_id=leaking.stable_sample_id,
            membership=leaking.membership,
            maps_by_depth=leaking.maps_by_depth,
        ),
    ]
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.fit_gt_map_calibration(
            samples,
            candidate_layers=target_fixture.candidate_layers,
            prediction_depths=target_fixture.prediction_depths,
        )
    assert _error_code(excinfo) == "B2_TARGET_CALIBRATION_LEAKAGE"


def test_gt_map_calibration_artifact_rejects_wrong_training_count(
    target_fixture: Any,
) -> None:
    calibration = subject.fit_gt_map_calibration(
        fixtures.fixture_calibration_samples(target_fixture)[:4],
        candidate_layers=target_fixture.candidate_layers,
        prediction_depths=target_fixture.prediction_depths,
    )
    hashes = {
        sample.stable_sample_id: sample.teacher_record_scientific_sha256
        for sample in target_fixture.by_membership("training")[:4]
    }
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.build_gt_map_calibration_artifact(
            calibration,
            source_teacher_record_scientific_sha256_by_id=hashes,
            teacher_cache_scientific_sha256=target_fixture.teacher_cache_scientific_sha256,
            teacher_cache_sample_coverage_sha256=(
                target_fixture.teacher_cache_sample_coverage_sha256
            ),
            descriptor_collection_scientific_sha256=(
                target_fixture.descriptor_collection_scientific_sha256
            ),
            split_scientific_sha256=target_fixture.split_scientific_sha256,
            checkpoint_sha256=target_fixture.checkpoint_sha256,
            execution_profile_sha256=target_fixture.execution_profile_sha256,
            expected_training_count=16,
            artifact_kind="test_fixture",
        )
    assert _error_code(excinfo) == "B2_TARGET_CALIBRATION_COUNT_MISMATCH"


def test_gt_map_calibration_artifact_rejects_teacher_hash_coverage_drift(
    target_fixture: Any,
) -> None:
    calibration = fixtures.fit_fixture_calibration(target_fixture)
    hashes = {
        sample.stable_sample_id: sample.teacher_record_scientific_sha256
        for sample in target_fixture.by_membership("training")
    }
    leaking = target_fixture.by_membership("calibration")[0]
    hashes[leaking.stable_sample_id] = leaking.teacher_record_scientific_sha256
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.build_gt_map_calibration_artifact(
            calibration,
            source_teacher_record_scientific_sha256_by_id=hashes,
            teacher_cache_scientific_sha256=target_fixture.teacher_cache_scientific_sha256,
            teacher_cache_sample_coverage_sha256=(
                target_fixture.teacher_cache_sample_coverage_sha256
            ),
            descriptor_collection_scientific_sha256=(
                target_fixture.descriptor_collection_scientific_sha256
            ),
            split_scientific_sha256=target_fixture.split_scientific_sha256,
            checkpoint_sha256=target_fixture.checkpoint_sha256,
            execution_profile_sha256=target_fixture.execution_profile_sha256,
            expected_training_count=16,
            artifact_kind="test_fixture",
        )
    assert _error_code(excinfo) == "B2_TARGET_CALIBRATION_COVERAGE_MISMATCH"


# ---------------------------------------------------------------------------
# Sample target record schema
# ---------------------------------------------------------------------------


def test_target_record_carries_both_families_at_every_depth(
    target_fixture: Any, target_records: Any
) -> None:
    assert len(target_records) == 32
    expected_coalitions = {12: 4, 18: 8, 24: 16}
    for record in target_records:
        assert record["schema_version"] == 1
        assert record["target_record_contract_version"] == subject.RECORD_CONTRACT_VERSION
        assert record["target_families"] == ["gt_localization", "teacher_fidelity"]
        assert record["candidate_layers"] == list(target_fixture.candidate_layers)
        assert record["prediction_depths"] == list(target_fixture.prediction_depths)
        assert record["statistics_dtype"] == "float64"
        assert sorted(record["depth_targets"], key=int) == ["12", "18", "24"]
        for depth in target_fixture.prediction_depths:
            block = _depth_block(record, depth)
            players = subject.players_for_depth(target_fixture.candidate_layers, depth)
            assert block["prediction_depth"] == depth
            assert block["ordered_player_layers"] == list(players)
            assert len(block["coalition_table"]) == expected_coalitions[depth]
            assert [int(entry["bitmask"]) for entry in block["coalition_table"]] == list(
                range(expected_coalitions[depth])
            )
            for family in subject.TARGET_FAMILIES:
                family_block = block[family]
                assert set(family_block["raw_signed_shapley_by_layer"]) == {
                    str(layer) for layer in players
                }
                assert set(family_block["positive_allocation_target_by_layer"]) == {
                    str(layer) for layer in players
                }
                assert family_block["efficiency_residual"] <= subject.EFFICIENCY_TOLERANCE
                allocation = family_block["positive_allocation_target_by_layer"].values()
                assert sum(allocation) == pytest.approx(1.0, abs=1e-12)
                assert all(value >= 0.0 for value in allocation)


def test_target_record_binds_every_upstream_and_contract_identity(
    target_fixture: Any, target_records: Any, calibration_artifact: Any
) -> None:
    for record in target_records:
        sample = target_fixture.sample_by_id(record["stable_sample_id"])
        assert record["split_membership"] == sample.membership
        assert record["category"] == sample.category
        assert record["label"] == sample.label
        assert record["anomaly_type"] == sample.anomaly_type
        assert record["coalition_contract_version"] == subject.COALITION_CONTRACT_VERSION
        assert record["utility_contract_version"] == subject.UTILITY_CONTRACT_VERSION
        assert record["shapley_contract_version"] == subject.SHAPLEY_CONTRACT_VERSION
        assert record["allocation_contract_version"] == subject.ALLOCATION_CONTRACT_VERSION
        assert record["gt_map_calibration_scientific_sha256"] == (
            calibration_artifact["gt_map_calibration_scientific_sha256"]
        )
        assert record["source_teacher_record_scientific_sha256"] == (
            sample.teacher_record_scientific_sha256
        )
        assert record["descriptor_record_scientific_sha256"] == (
            sample.descriptor_record["descriptor_record_scientific_sha256"]
        )
        assert record["teacher_cache_scientific_sha256"] == (
            target_fixture.teacher_cache_scientific_sha256
        )
        assert record["teacher_cache_sample_coverage_sha256"] == (
            target_fixture.teacher_cache_sample_coverage_sha256
        )
        assert record["descriptor_collection_scientific_sha256"] == (
            target_fixture.descriptor_collection_scientific_sha256
        )
        assert record["split_scientific_sha256"] == target_fixture.split_scientific_sha256
        assert record["checkpoint_sha256"] == target_fixture.checkpoint_sha256
        assert record["execution_profile_sha256"] == target_fixture.execution_profile_sha256


def test_target_record_mask_and_teacher_reference_provenance(
    target_fixture: Any, target_records: Any
) -> None:
    for record in target_records:
        sample = target_fixture.sample_by_id(record["stable_sample_id"])
        mask = record["mask_provenance"]
        assert mask["binarization_threshold"] == 0.5
        assert mask["alignment_mode"] == "nearest"
        assert mask["mask_shape"] == [
            int(target_fixture.map_shape[-2]),
            int(target_fixture.map_shape[-1]),
        ]
        assert mask["mask_identity"] == sample.mask_identity
        assert len(mask["mask_digest"]) == 64
        if sample.is_anomalous:
            assert mask["mask_source"] == "production_gt_mask"
            assert mask["positive_pixel_count"] > 0
            assert mask["background_pixel_count"] > 0
        else:
            assert mask["mask_source"] == "normal_all_zero_mask"
            assert mask["positive_pixel_count"] == 0
        teacher = record["teacher_reference_provenance"]
        assert teacher["fusion_function"] == "rad.models.dlcm.sum_preserving_fusion"
        assert teacher["reconstruction_verified"] is True
        assert teacher["source_candidate_layers"] == list(target_fixture.candidate_layers)
        assert teacher["cached_full_depth_map_digest"] == subject.full_depth_map_digest(
            sample.full_depth_map
        )
        assert teacher["full_depth_map_dtype"] == "float32"


def test_target_record_coalition_table_has_components_but_no_maps(
    target_records: Any,
) -> None:
    record = _record_for(target_records, "training", label=1)
    entry = _coalition_entry(record, 24, 15)
    assert entry["layer_ids"] == [6, 12, 18, 24]
    assert entry["coalition_size"] == 4
    assert set(entry["gt_localization"]["utility_components"]) == {
        "pixel_ap",
        "soft_dice",
        "background_penalty",
        "background_pixel_count",
        "background_top1_percent_k",
        "background_top1_percent_mean",
        "background_global_mean",
    }
    assert set(entry["teacher_fidelity"]["utility_components"]) == {
        "spearman_raw",
        "spearman_fidelity",
        "top1_overlap",
    }
    for path, value in _walk(record):
        assert not isinstance(value, torch.Tensor), path
        assert isinstance(
            value, dict | list | tuple | str | int | float | bool | type(None)
        ), path
    keys = {path.rsplit(".", 1)[-1] for path, _ in _walk(record) if path}
    assert not any("map" == key or key.endswith("_map") for key in keys)


def test_target_record_normal_samples_use_the_suppression_utility(
    target_records: Any,
) -> None:
    record = _record_for(target_records, "training", label=0)
    block = _depth_block(record, 12)
    assert block["gt_localization"]["utility_mode"] == "normal"
    for entry in block["coalition_table"]:
        assert set(entry["gt_localization"]["utility_components"]) == {
            "top1_percent_k",
            "top1_percent_mean",
            "global_mean",
        }
    anomalous = _record_for(target_records, "training", label=1)
    assert _depth_block(anomalous, 12)["gt_localization"]["utility_mode"] == "abnormal"


def test_target_record_values_are_float64_and_never_standardized(
    target_records: Any,
) -> None:
    numeric_keys = {
        "raw_utility",
        "centered_value",
        "efficiency_residual",
        "empty_coalition_raw_utility",
        "grand_coalition_centered_value",
    }
    for record in target_records:
        for path, value in _walk(record):
            key = path.rsplit(".", 1)[-1]
            assert "standardized" not in key
            assert "z_score" not in key
            if key in numeric_keys:
                assert isinstance(value, float), path
        for depth_block in record["depth_targets"].values():
            for family in subject.TARGET_FAMILIES:
                for mapping_key in (
                    "raw_signed_shapley_by_layer",
                    "positive_allocation_target_by_layer",
                ):
                    for value in depth_block[family][mapping_key].values():
                        assert isinstance(value, float)
                        assert math.isfinite(value)


def test_target_record_mathematics_match_the_story_one_primitives(
    target_fixture: Any, target_records: Any, calibration_artifact: Any
) -> None:
    record = _record_for(target_records, "training", label=1)
    sample = target_fixture.sample_by_id(record["stable_sample_id"])
    depth = 12
    players = subject.players_for_depth(target_fixture.candidate_layers, depth)
    bounds = calibration_artifact["by_depth"][str(depth)]
    mask = subject.binarize_and_validate_mask(
        sample.mask,
        is_anomalous=True,
        map_shape=(target_fixture.map_shape[-2], target_fixture.map_shape[-1]),
    )
    layer_maps = sample.maps_by_depth[depth]
    raw_gt: dict[int, float] = {}
    raw_teacher: dict[int, float] = {}
    for coalition in subject.enumerate_coalitions(players):
        fused = subject.fuse_equal_average(
            layer_maps, coalition.layer_ids, template=layer_maps[players[0]]
        )
        calibrated = subject.apply_gt_calibration(fused, bounds["q_low"], bounds["q_high"])
        raw_gt[coalition.bitmask] = subject.gt_utility_abnormal(
            raw_map=fused, calibrated_map=calibrated, mask=mask
        ).utility
        raw_teacher[coalition.bitmask] = subject.teacher_utility(
            fused, sample.full_depth_map
        ).utility
    for family, raw in (("gt_localization", raw_gt), ("teacher_fidelity", raw_teacher)):
        centered = subject.center_utilities(raw)
        phi = subject.exact_shapley(players, centered)
        allocation = subject.positive_allocation(phi)
        block = _depth_block(record, depth)[family]
        for layer in players:
            assert block["raw_signed_shapley_by_layer"][str(layer)] == pytest.approx(
                phi[layer], abs=1e-15
            )
            assert block["positive_allocation_target_by_layer"][str(layer)] == (
                pytest.approx(allocation[layer], abs=1e-15)
            )
        for bitmask, value in raw.items():
            entry = _coalition_entry(record, depth, bitmask)
            assert entry[family]["raw_utility"] == pytest.approx(value, abs=1e-15)
            assert entry[family]["centered_value"] == pytest.approx(
                centered[bitmask], abs=1e-15
            )
        assert block["grand_coalition_centered_value"] == pytest.approx(
            centered[(1 << len(players)) - 1], abs=1e-15
        )


def test_target_record_scientific_hash_excludes_paths_timestamps_and_file_hashes(
    target_records: Any,
) -> None:
    record = copy.deepcopy(target_records[0])
    claimed = record["contribution_target_record_scientific_sha256"]
    assert subject.contribution_target_record_scientific_sha256(record) == claimed
    for key, value in (
        ("absolute_output_path", "/tmp/records/x.pt"),
        ("relative_record_path", "records/x.pt"),
        ("record_file_sha256", "0" * 64),
        ("timestamp", "2026-07-29T00:00:00Z"),
        ("git_branch", "phase-b2-contribution-target-contract"),
        ("worktree_path", "/root/autodl-tmp"),
        ("runtime_attestation_sha256", "f" * 64),
    ):
        polluted = dict(record)
        polluted[key] = value
        assert subject.contribution_target_record_scientific_sha256(polluted) == claimed
    polluted = dict(record)
    polluted["undeclared_science"] = "leak"
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.contribution_target_record_scientific_sha256(polluted)
    assert _error_code(excinfo) == "B2_TARGET_RECORD_HASH_SCHEMA_INVALID"


def test_target_record_scientific_hash_requires_every_whitelisted_field(
    target_records: Any,
) -> None:
    for field in ("depth_targets", "mask_provenance", "split_membership"):
        broken = copy.deepcopy(target_records[0])
        del broken[field]
        with pytest.raises(subject.ContributionTargetError) as excinfo:
            subject.contribution_target_record_scientific_sha256(broken)
        assert _error_code(excinfo) == "B2_TARGET_RECORD_HASH_SCHEMA_INVALID"


def test_validate_contribution_target_record_detects_content_drift(
    target_fixture: Any, target_records: Any
) -> None:
    record = copy.deepcopy(target_records[0])
    assert (
        subject.validate_contribution_target_record(
            record,
            candidate_layers=target_fixture.candidate_layers,
            prediction_depths=target_fixture.prediction_depths,
        )
        is None
    )
    drifted = copy.deepcopy(record)
    block = _depth_block(drifted, 12)["gt_localization"]
    layer = next(iter(block["raw_signed_shapley_by_layer"]))
    block["raw_signed_shapley_by_layer"][layer] += 0.5
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.validate_contribution_target_record(
            drifted,
            candidate_layers=target_fixture.candidate_layers,
            prediction_depths=target_fixture.prediction_depths,
        )
    assert _error_code(excinfo) == "B2_TARGET_RECORD_HASH_MISMATCH"

    missing_depth = copy.deepcopy(record)
    del missing_depth["depth_targets"]["18"]
    missing_depth["contribution_target_record_scientific_sha256"] = (
        subject.contribution_target_record_scientific_sha256(missing_depth)
    )
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.validate_contribution_target_record(
            missing_depth,
            candidate_layers=target_fixture.candidate_layers,
            prediction_depths=target_fixture.prediction_depths,
        )
    assert _error_code(excinfo) == "B2_TARGET_RECORD_DEPTH_MISSING"


def test_build_target_record_rejects_unverified_teacher_reference(
    target_fixture: Any, calibration_artifact: Any
) -> None:
    sample = target_fixture.samples[0]
    provenance = subject.TeacherReferenceProvenance(
        cached_full_depth_map_digest=subject.full_depth_map_digest(sample.full_depth_map),
        reconstruction_verified=False,
        source_candidate_layers=target_fixture.candidate_layers,
    )
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.build_contribution_target_record(
            sample=subject.ContributionTargetSample(
                stable_sample_id=sample.stable_sample_id,
                split_membership=sample.membership,
                category=sample.category,
                label=sample.label,
                anomaly_type=sample.anomaly_type,
                maps_by_depth=sample.maps_by_depth,
                mask=sample.mask,
                teacher_reference_map=sample.full_depth_map,
            ),
            calibration_artifact=calibration_artifact,
            upstream=fixtures.fixture_upstream(target_fixture, sample),
            mask_provenance=fixtures.fixture_mask_provenance(sample),
            teacher_reference_provenance=provenance,
            candidate_layers=target_fixture.candidate_layers,
            prediction_depths=target_fixture.prediction_depths,
            artifact_kind="test_fixture",
        )
    assert _error_code(excinfo) == "B2_TARGET_TEACHER_REFERENCE_UNVERIFIED"


def test_build_target_record_rejects_teacher_reference_digest_drift(
    target_fixture: Any, calibration_artifact: Any
) -> None:
    sample = target_fixture.samples[0]
    other = target_fixture.samples[1]
    provenance = subject.TeacherReferenceProvenance(
        cached_full_depth_map_digest=subject.full_depth_map_digest(other.full_depth_map),
        reconstruction_verified=True,
        source_candidate_layers=target_fixture.candidate_layers,
    )
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.build_contribution_target_record(
            sample=subject.ContributionTargetSample(
                stable_sample_id=sample.stable_sample_id,
                split_membership=sample.membership,
                category=sample.category,
                label=sample.label,
                anomaly_type=sample.anomaly_type,
                maps_by_depth=sample.maps_by_depth,
                mask=sample.mask,
                teacher_reference_map=sample.full_depth_map,
            ),
            calibration_artifact=calibration_artifact,
            upstream=fixtures.fixture_upstream(target_fixture, sample),
            mask_provenance=fixtures.fixture_mask_provenance(sample),
            teacher_reference_provenance=provenance,
            candidate_layers=target_fixture.candidate_layers,
            prediction_depths=target_fixture.prediction_depths,
            artifact_kind="test_fixture",
        )
    assert _error_code(excinfo) == "B2_TARGET_TEACHER_REFERENCE_DIGEST_MISMATCH"


def _build_record_with(
    target_fixture: Any,
    calibration_artifact: Any,
    sample: Any,
    **overrides: Any,
) -> Any:
    payload = {
        "stable_sample_id": sample.stable_sample_id,
        "split_membership": sample.membership,
        "category": sample.category,
        "label": sample.label,
        "anomaly_type": sample.anomaly_type,
        "maps_by_depth": sample.maps_by_depth,
        "mask": sample.mask,
        "teacher_reference_map": sample.full_depth_map,
    }
    payload.update(overrides)
    return subject.build_contribution_target_record(
        sample=subject.ContributionTargetSample(**payload),
        calibration_artifact=calibration_artifact,
        upstream=fixtures.fixture_upstream(target_fixture, sample),
        mask_provenance=fixtures.fixture_mask_provenance(sample),
        teacher_reference_provenance=fixtures.fixture_teacher_reference_provenance(
            target_fixture, sample
        ),
        candidate_layers=target_fixture.candidate_layers,
        prediction_depths=target_fixture.prediction_depths,
        artifact_kind="test_fixture",
    )


def test_build_target_record_rejects_missing_depth_maps(
    target_fixture: Any, calibration_artifact: Any
) -> None:
    sample = target_fixture.samples[0]
    truncated = {
        depth: maps
        for depth, maps in sample.maps_by_depth.items()
        if depth != max(target_fixture.prediction_depths)
    }
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        _build_record_with(
            target_fixture, calibration_artifact, sample, maps_by_depth=truncated
        )
    assert _error_code(excinfo) == "B2_TARGET_RECORD_DEPTH_MISSING"

    incomplete = dict(sample.maps_by_depth)
    incomplete[12] = {6: sample.maps_by_depth[12][6]}
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        _build_record_with(
            target_fixture, calibration_artifact, sample, maps_by_depth=incomplete
        )
    assert _error_code(excinfo) == "B2_TARGET_RECORD_LAYER_SET_INVALID"


def test_build_target_record_rejects_wrong_dtype_and_nonfinite_maps(
    target_fixture: Any, calibration_artifact: Any
) -> None:
    sample = target_fixture.samples[0]
    integral = dict(sample.maps_by_depth)
    integral[12] = dict(sample.maps_by_depth[12])
    integral[12][6] = sample.maps_by_depth[12][6].to(dtype=torch.int64)
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        _build_record_with(
            target_fixture, calibration_artifact, sample, maps_by_depth=integral
        )
    assert _error_code(excinfo) == "B2_TARGET_MAP_DTYPE_INVALID"

    nonfinite = dict(sample.maps_by_depth)
    nonfinite[12] = dict(sample.maps_by_depth[12])
    broken = sample.maps_by_depth[12][6].clone()
    broken.reshape(-1)[0] = float("nan")
    nonfinite[12][6] = broken
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        _build_record_with(
            target_fixture, calibration_artifact, sample, maps_by_depth=nonfinite
        )
    assert _error_code(excinfo) == "B2_TARGET_MAP_NONFINITE"


def test_build_target_record_rejects_mask_label_disagreement(
    target_fixture: Any, calibration_artifact: Any
) -> None:
    anomalous = _first_sample(target_fixture, label=1)
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        _build_record_with(
            target_fixture,
            calibration_artifact,
            anomalous,
            mask=torch.zeros(target_fixture.map_shape, dtype=torch.float32),
        )
    assert _error_code(excinfo) == "B2_TARGET_MASK_ANOMALY_MISSING"

    normal = _first_sample(target_fixture, label=0)
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.build_contribution_target_record(
            sample=subject.ContributionTargetSample(
                stable_sample_id=normal.stable_sample_id,
                split_membership=normal.membership,
                category=normal.category,
                label=normal.label,
                anomaly_type=normal.anomaly_type,
                maps_by_depth=normal.maps_by_depth,
                mask=torch.ones(target_fixture.map_shape, dtype=torch.float32),
                teacher_reference_map=normal.full_depth_map,
            ),
            calibration_artifact=calibration_artifact,
            upstream=fixtures.fixture_upstream(target_fixture, normal),
            mask_provenance=fixtures.fixture_mask_provenance(normal),
            teacher_reference_provenance=fixtures.fixture_teacher_reference_provenance(
                target_fixture, normal
            ),
            candidate_layers=target_fixture.candidate_layers,
            prediction_depths=target_fixture.prediction_depths,
            artifact_kind="test_fixture",
        )
    assert _error_code(excinfo) == "B2_TARGET_MASK_NORMAL_NOT_EMPTY"


def _first_sample(target_fixture: Any, *, label: int) -> Any:
    for sample in target_fixture.samples:
        if sample.label == label:
            return sample
    raise AssertionError(f"no fixture sample with label {label}")


def test_build_target_record_rejects_inconsistent_mask_provenance(
    target_fixture: Any, calibration_artifact: Any
) -> None:
    sample = _first_sample(target_fixture, label=1)
    for provenance, code in (
        (
            subject.MaskProvenance(
                mask_identity=sample.mask_identity,
                mask_source="normal_all_zero_mask",
            ),
            "B2_TARGET_MASK_PROVENANCE_INVALID",
        ),
        (
            subject.MaskProvenance(
                mask_identity=sample.mask_identity,
                mask_source="production_gt_mask",
                alignment_mode="bilinear",
            ),
            "B2_TARGET_MASK_PROVENANCE_INVALID",
        ),
        (
            subject.MaskProvenance(
                mask_identity=sample.mask_identity,
                mask_source="production_gt_mask",
                binarization_threshold=0.25,
            ),
            "B2_TARGET_MASK_PROVENANCE_INVALID",
        ),
    ):
        with pytest.raises(subject.ContributionTargetError) as excinfo:
            subject.build_contribution_target_record(
                sample=subject.ContributionTargetSample(
                    stable_sample_id=sample.stable_sample_id,
                    split_membership=sample.membership,
                    category=sample.category,
                    label=sample.label,
                    anomaly_type=sample.anomaly_type,
                    maps_by_depth=sample.maps_by_depth,
                    mask=sample.mask,
                    teacher_reference_map=sample.full_depth_map,
                ),
                calibration_artifact=calibration_artifact,
                upstream=fixtures.fixture_upstream(target_fixture, sample),
                mask_provenance=provenance,
                teacher_reference_provenance=(
                    fixtures.fixture_teacher_reference_provenance(target_fixture, sample)
                ),
                candidate_layers=target_fixture.candidate_layers,
                prediction_depths=target_fixture.prediction_depths,
                artifact_kind="test_fixture",
            )
        assert _error_code(excinfo) == code


def test_build_target_record_rejects_calibration_artifact_drift(
    target_fixture: Any, calibration_artifact: Any
) -> None:
    sample = target_fixture.samples[0]
    tampered = copy.deepcopy(calibration_artifact)
    tampered["by_depth"]["18"]["q_low"] = float(tampered["by_depth"]["18"]["q_low"]) - 1.0
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.build_contribution_target_record(
            sample=subject.ContributionTargetSample(
                stable_sample_id=sample.stable_sample_id,
                split_membership=sample.membership,
                category=sample.category,
                label=sample.label,
                anomaly_type=sample.anomaly_type,
                maps_by_depth=sample.maps_by_depth,
                mask=sample.mask,
                teacher_reference_map=sample.full_depth_map,
            ),
            calibration_artifact=tampered,
            upstream=fixtures.fixture_upstream(target_fixture, sample),
            mask_provenance=fixtures.fixture_mask_provenance(sample),
            teacher_reference_provenance=fixtures.fixture_teacher_reference_provenance(
                target_fixture, sample
            ),
            candidate_layers=target_fixture.candidate_layers,
            prediction_depths=target_fixture.prediction_depths,
            artifact_kind="test_fixture",
        )
    assert _error_code(excinfo) == "B2_TARGET_CALIBRATION_HASH_MISMATCH"


def test_build_target_record_fails_closed_on_efficiency_violation(
    target_fixture: Any,
    calibration_artifact: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = target_fixture.samples[0]
    original = subject.exact_shapley

    def _perturbed(players: Any, centered: Any) -> Any:
        phi = dict(original(players, centered))
        phi[players[0]] = phi[players[0]] + 1.0
        return phi

    monkeypatch.setattr(subject, "exact_shapley", _perturbed)
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        _build_record_with(target_fixture, calibration_artifact, sample)
    assert _error_code(excinfo) == "B2_TARGET_SHAPLEY_EFFICIENCY_VIOLATION"


def test_build_target_record_rejects_unknown_split_membership(
    target_fixture: Any, calibration_artifact: Any
) -> None:
    sample = target_fixture.samples[0]
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        _build_record_with(
            target_fixture, calibration_artifact, sample, split_membership="target"
        )
    assert _error_code(excinfo) == "B2_TARGET_RECORD_MEMBERSHIP_INVALID"


# ---------------------------------------------------------------------------
# Shapley normalization artifact
# ---------------------------------------------------------------------------


def test_shapley_normalization_axes_are_family_depth_layer(
    target_fixture: Any, target_records: Any, normalization: Any
) -> None:
    assert normalization["normalization_contract_version"] == (
        subject.NORMALIZATION_CONTRACT_VERSION
    )
    assert normalization["statistics_dtype"] == "float64"
    assert normalization["standard_deviation_ddof"] == 0
    assert normalization["target_families"] == list(subject.TARGET_FAMILIES)
    training = fixtures.records_by_membership(target_records, "training")
    ordered_ids = sorted(row["stable_sample_id"] for row in training)
    assert normalization["ordered_training_stable_sample_ids"] == ordered_ids
    assert set(normalization["contribution_target_record_scientific_sha256_by_id"]) == set(
        ordered_ids
    )
    for family in subject.TARGET_FAMILIES:
        for depth in target_fixture.prediction_depths:
            players = subject.players_for_depth(target_fixture.candidate_layers, depth)
            entry = normalization["axes"][family][str(depth)]
            assert entry["prediction_depth"] == depth
            assert [layer["candidate_layer_id"] for layer in entry["layers"]] == list(players)
            for layer in entry["layers"]:
                values = [
                    float(
                        row["depth_targets"][str(depth)][family][
                            "raw_signed_shapley_by_layer"
                        ][str(layer["candidate_layer_id"])]
                    )
                    for row in sorted(training, key=lambda item: item["stable_sample_id"])
                ]
                mean = math.fsum(values) / len(values)
                variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
                assert layer["count"] == 16
                assert layer["mean"] == pytest.approx(mean, abs=1e-15)
                assert layer["std"] == pytest.approx(math.sqrt(variance), abs=1e-15)
                assert layer["minimum"] == pytest.approx(min(values), abs=1e-15)
                assert layer["maximum"] == pytest.approx(max(values), abs=1e-15)
                assert layer["zero_variance"] is (layer["std"] == 0.0)


def test_shapley_normalization_uses_only_the_sixteen_training_records(
    target_fixture: Any, target_records: Any
) -> None:
    for membership in ("calibration", "evaluation"):
        leaking = [
            *fixtures.records_by_membership(target_records, "training")[:15],
            fixtures.records_by_membership(target_records, membership)[0],
        ]
        with pytest.raises(subject.ContributionTargetError) as excinfo:
            subject.compute_shapley_normalization(
                leaking,
                candidate_layers=target_fixture.candidate_layers,
                prediction_depths=target_fixture.prediction_depths,
                expected_training_count=16,
                artifact_kind="test_fixture",
            )
        assert _error_code(excinfo) == "B2_TARGET_NORMALIZATION_MEMBERSHIP_INVALID"

    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.compute_shapley_normalization(
            fixtures.records_by_membership(target_records, "training")[:8],
            candidate_layers=target_fixture.candidate_layers,
            prediction_depths=target_fixture.prediction_depths,
            expected_training_count=16,
            artifact_kind="test_fixture",
        )
    assert _error_code(excinfo) == "B2_TARGET_NORMALIZATION_COUNT_MISMATCH"


def test_shapley_normalization_is_input_order_independent(
    target_fixture: Any, target_records: Any, normalization: Any
) -> None:
    training = list(fixtures.records_by_membership(target_records, "training"))
    shuffled = [*training[8:], *training[:8]]
    recomputed = subject.compute_shapley_normalization(
        shuffled,
        candidate_layers=target_fixture.candidate_layers,
        prediction_depths=target_fixture.prediction_depths,
        expected_training_count=16,
        artifact_kind="test_fixture",
    )
    assert recomputed["shapley_normalization_scientific_sha256"] == (
        normalization["shapley_normalization_scientific_sha256"]
    )
    assert recomputed["ordered_training_stable_sample_ids"] == (
        normalization["ordered_training_stable_sample_ids"]
    )


def test_shapley_normalization_hash_excludes_non_scientific_fields(
    normalization: Any,
) -> None:
    artifact = copy.deepcopy(normalization)
    claimed = artifact["shapley_normalization_scientific_sha256"]
    assert subject.shapley_normalization_scientific_sha256(artifact) == claimed
    for key, value in (
        ("absolute_output_path", "/tmp/shapley_normalization.pt"),
        ("timestamp", "2026-07-29T00:00:00Z"),
        ("normalization_file_sha256", "0" * 64),
    ):
        polluted = dict(artifact)
        polluted[key] = value
        assert subject.shapley_normalization_scientific_sha256(polluted) == claimed
    polluted = dict(artifact)
    polluted["undeclared_statistic"] = 3
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.shapley_normalization_scientific_sha256(polluted)
    assert _error_code(excinfo) == "B2_TARGET_NORMALIZATION_HASH_SCHEMA_INVALID"


def test_standardize_signed_shapley_is_a_read_time_z_score(normalization: Any) -> None:
    artifact = copy.deepcopy(normalization)
    entry = artifact["axes"]["gt_localization"]["12"]["layers"][0]
    mean = float(entry["mean"])
    std = float(entry["std"])
    assert std > 0.0
    value = mean + 2.0 * std
    assert subject.standardize_signed_shapley(
        value,
        artifact,
        target_family="gt_localization",
        prediction_depth=12,
        candidate_layer_id=int(entry["candidate_layer_id"]),
    ) == pytest.approx(2.0, abs=1e-12)

    entry["std"] = 0.0
    entry["zero_variance"] = True
    assert subject.standardize_signed_shapley(
        value,
        artifact,
        target_family="gt_localization",
        prediction_depth=12,
        candidate_layer_id=int(entry["candidate_layer_id"]),
    ) == 0.0


def test_standardize_signed_shapley_rejects_unknown_axes(normalization: Any) -> None:
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.standardize_signed_shapley(
            0.0,
            normalization,
            target_family="residual_gain",
            prediction_depth=12,
            candidate_layer_id=6,
        )
    assert _error_code(excinfo) == "B2_TARGET_FAMILY_INVALID"
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.standardize_signed_shapley(
            0.0,
            normalization,
            target_family="gt_localization",
            prediction_depth=12,
            candidate_layer_id=18,
        )
    assert _error_code(excinfo) == "B2_TARGET_NORMALIZATION_AXIS_MISSING"


# ---------------------------------------------------------------------------
# Coverage, collection, and plan identities
# ---------------------------------------------------------------------------


def test_sample_coverage_hash_binds_ordered_ids_and_split_counts(
    target_records: Any,
) -> None:
    coverage = subject.contribution_target_sample_coverage_sha256(target_records)
    assert len(coverage) == 64
    shuffled = [*target_records[16:], *target_records[:16]]
    assert subject.contribution_target_sample_coverage_sha256(shuffled) == coverage
    drifted = copy.deepcopy(list(target_records))
    drifted[0]["split_membership"] = "calibration"
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.contribution_target_sample_coverage_sha256(drifted)
    assert _error_code(excinfo) == "B2_TARGET_COVERAGE_COUNT_MISMATCH"
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.contribution_target_sample_coverage_sha256(target_records[:31])
    assert _error_code(excinfo) == "B2_TARGET_COVERAGE_COUNT_MISMATCH"


def test_split_coverage_hashes_are_distinct_and_reject_foreign_records(
    target_records: Any,
) -> None:
    training = fixtures.records_by_membership(target_records, "training")
    calibration = fixtures.records_by_membership(target_records, "calibration")
    evaluation = fixtures.records_by_membership(target_records, "evaluation")
    hashes = {
        subject.training_target_coverage_sha256(training),
        subject.calibration_target_coverage_sha256(calibration),
        subject.evaluation_target_coverage_sha256(evaluation),
    }
    assert len(hashes) == 3
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.training_target_coverage_sha256(calibration)
    assert _error_code(excinfo) == "B2_TARGET_COVERAGE_MEMBERSHIP_INVALID"
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.calibration_target_coverage_sha256(calibration[:4])
    assert _error_code(excinfo) == "B2_TARGET_COVERAGE_COUNT_MISMATCH"
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.evaluation_target_coverage_sha256(training)
    assert _error_code(excinfo) == "B2_TARGET_COVERAGE_MEMBERSHIP_INVALID"


def test_collection_identity_binds_records_calibration_and_normalization(
    target_fixture: Any,
    target_records: Any,
    calibration_artifact: Any,
    normalization: Any,
) -> None:
    baseline = subject.contribution_target_collection_scientific_sha256(
        records=target_records,
        calibration_artifact=calibration_artifact,
        normalization=normalization,
        candidate_layers=target_fixture.candidate_layers,
        prediction_depths=target_fixture.prediction_depths,
    )
    assert len(baseline) == 64
    shuffled = [*target_records[8:], *target_records[:8]]
    assert (
        subject.contribution_target_collection_scientific_sha256(
            records=shuffled,
            calibration_artifact=calibration_artifact,
            normalization=normalization,
            candidate_layers=target_fixture.candidate_layers,
            prediction_depths=target_fixture.prediction_depths,
        )
        == baseline
    )
    drifted_calibration = copy.deepcopy(calibration_artifact)
    drifted_calibration["gt_map_calibration_scientific_sha256"] = "1" * 64
    assert (
        subject.contribution_target_collection_scientific_sha256(
            records=target_records,
            calibration_artifact=drifted_calibration,
            normalization=normalization,
            candidate_layers=target_fixture.candidate_layers,
            prediction_depths=target_fixture.prediction_depths,
        )
        != baseline
    )
    drifted_normalization = copy.deepcopy(normalization)
    drifted_normalization["shapley_normalization_scientific_sha256"] = "2" * 64
    assert (
        subject.contribution_target_collection_scientific_sha256(
            records=target_records,
            calibration_artifact=drifted_normalization and drifted_calibration,
            normalization=drifted_normalization,
            candidate_layers=target_fixture.candidate_layers,
            prediction_depths=target_fixture.prediction_depths,
        )
        != baseline
    )


def test_contribution_plan_binds_all_seven_identities(
    target_fixture: Any,
    target_records: Any,
    calibration_artifact: Any,
    normalization: Any,
) -> None:
    plan = subject.build_contribution_plan(
        records=target_records,
        calibration_artifact=calibration_artifact,
        normalization=normalization,
        candidate_layers=target_fixture.candidate_layers,
        prediction_depths=target_fixture.prediction_depths,
    )
    identities = (
        "gt_map_calibration_scientific_sha256",
        "contribution_target_sample_coverage_sha256",
        "contribution_target_collection_scientific_sha256",
        "shapley_normalization_scientific_sha256",
        "training_target_coverage_sha256",
        "calibration_target_coverage_sha256",
        "evaluation_target_coverage_sha256",
    )
    for identity in identities:
        assert len(plan[identity]) == 64
    assert len(set(plan[identity] for identity in identities)) == 7
    assert len(plan["contribution_plan_scientific_sha256"]) == 64
    assert plan["teacher_forward_count"] == 0
    assert plan["official_materialization_enabled"] is False
    assert plan["planned_record_count"] == 32
    assert plan["planned_split_counts"] == {
        "training": 16,
        "calibration": 8,
        "evaluation": 8,
    }
    assert plan["planned_ordered_stable_sample_ids"] == sorted(
        row["stable_sample_id"] for row in target_records
    )
    assert plan["prediction_depths"] == list(target_fixture.prediction_depths)
    assert plan["candidate_layers"] == list(target_fixture.candidate_layers)
    assert plan["contract_versions"] == {
        "coalition": subject.COALITION_CONTRACT_VERSION,
        "utility": subject.UTILITY_CONTRACT_VERSION,
        "shapley": subject.SHAPLEY_CONTRACT_VERSION,
        "allocation": subject.ALLOCATION_CONTRACT_VERSION,
        "record": subject.RECORD_CONTRACT_VERSION,
        "calibration": subject.CALIBRATION_CONTRACT_VERSION,
        "normalization": subject.NORMALIZATION_CONTRACT_VERSION,
        "collection": subject.COLLECTION_CONTRACT_VERSION,
        "plan": subject.PLAN_CONTRACT_VERSION,
    }
    # A pure in-memory plan: recomputing it twice is identical and writes nothing.
    again = subject.build_contribution_plan(
        records=[*target_records[4:], *target_records[:4]],
        calibration_artifact=calibration_artifact,
        normalization=normalization,
        candidate_layers=target_fixture.candidate_layers,
        prediction_depths=target_fixture.prediction_depths,
    )
    assert again["contribution_plan_scientific_sha256"] == (
        plan["contribution_plan_scientific_sha256"]
    )


@pytest.mark.parametrize(
    "mutate",
    ["record", "calibration", "normalization", "official_flag"],
)
def test_contribution_plan_hash_changes_when_any_bound_identity_changes(
    target_fixture: Any,
    target_records: Any,
    calibration_artifact: Any,
    normalization: Any,
    mutate: str,
) -> None:
    baseline = subject.build_contribution_plan(
        records=target_records,
        calibration_artifact=calibration_artifact,
        normalization=normalization,
        candidate_layers=target_fixture.candidate_layers,
        prediction_depths=target_fixture.prediction_depths,
    )["contribution_plan_scientific_sha256"]
    records = copy.deepcopy(list(target_records))
    artifact = copy.deepcopy(calibration_artifact)
    statistics = copy.deepcopy(normalization)
    official = False
    if mutate == "record":
        records[0]["contribution_target_record_scientific_sha256"] = "3" * 64
    elif mutate == "calibration":
        artifact["gt_map_calibration_scientific_sha256"] = "4" * 64
    elif mutate == "normalization":
        statistics["shapley_normalization_scientific_sha256"] = "5" * 64
    else:
        official = True
    mutated = subject.build_contribution_plan(
        records=records,
        calibration_artifact=artifact,
        normalization=statistics,
        candidate_layers=target_fixture.candidate_layers,
        prediction_depths=target_fixture.prediction_depths,
        official_materialization_enabled=official,
    )["contribution_plan_scientific_sha256"]
    assert mutated != baseline


def test_contribution_plan_rejects_wrong_split_counts(
    target_fixture: Any,
    target_records: Any,
    calibration_artifact: Any,
    normalization: Any,
) -> None:
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.build_contribution_plan(
            records=target_records[:24],
            calibration_artifact=calibration_artifact,
            normalization=normalization,
            candidate_layers=target_fixture.candidate_layers,
            prediction_depths=target_fixture.prediction_depths,
        )
    assert _error_code(excinfo) == "B2_TARGET_COVERAGE_COUNT_MISMATCH"


# ---------------------------------------------------------------------------
# Leakage-access helpers
# ---------------------------------------------------------------------------


def test_training_access_mode_fails_closed_on_calibration_or_evaluation_records(
    target_records: Any, normalization: Any
) -> None:
    training = list(fixtures.records_by_membership(target_records, "training"))
    for membership in ("calibration", "evaluation"):
        leaking = [
            *training,
            fixtures.records_by_membership(target_records, membership)[0],
        ]
        with pytest.raises(subject.ContributionTargetError) as excinfo:
            subject.load_targets_for_access(
                leaking, access_mode="training_only", normalization=normalization
            )
        assert _error_code(excinfo) == "B2_TARGET_ACCESS_LEAKAGE"


def test_access_modes_only_admit_their_own_split(
    target_records: Any, normalization: Any
) -> None:
    for access_mode, membership in (
        ("training_only", "training"),
        ("calibration_only", "calibration"),
        ("evaluation_only", "evaluation"),
    ):
        records = fixtures.records_by_membership(target_records, membership)
        views = subject.load_targets_for_access(
            records, access_mode=access_mode, normalization=normalization
        )
        assert [view["split_membership"] for view in views] == [membership] * len(records)
        assert [view["stable_sample_id"] for view in views] == sorted(
            row["stable_sample_id"] for row in records
        )
        foreign = fixtures.records_by_membership(
            target_records, "training" if membership != "training" else "evaluation"
        )
        with pytest.raises(subject.ContributionTargetError) as excinfo:
            subject.load_targets_for_access(
                foreign, access_mode=access_mode, normalization=normalization
            )
        assert _error_code(excinfo) == "B2_TARGET_ACCESS_LEAKAGE"


def test_access_mode_rejects_unknown_mode_and_tampered_normalization(
    target_records: Any, normalization: Any
) -> None:
    training = fixtures.records_by_membership(target_records, "training")
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.load_targets_for_access(
            training, access_mode="everything", normalization=normalization
        )
    assert _error_code(excinfo) == "B2_TARGET_ACCESS_MODE_INVALID"
    tampered = copy.deepcopy(normalization)
    tampered["axes"]["gt_localization"]["12"]["layers"][0]["mean"] = 12345.0
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.load_targets_for_access(
            training, access_mode="training_only", normalization=tampered
        )
    assert _error_code(excinfo) == "B2_TARGET_NORMALIZATION_HASH_MISMATCH"


def test_access_views_expose_raw_signed_allocation_and_standardized_values(
    target_fixture: Any, target_records: Any, normalization: Any
) -> None:
    training = fixtures.records_by_membership(target_records, "training")
    views = subject.load_targets_for_access(
        training, access_mode="training_only", normalization=normalization
    )
    by_id = {row["stable_sample_id"]: row for row in training}
    for view in views:
        record = by_id[view["stable_sample_id"]]
        for depth in target_fixture.prediction_depths:
            players = subject.players_for_depth(target_fixture.candidate_layers, depth)
            for family in subject.TARGET_FAMILIES:
                source = record["depth_targets"][str(depth)][family]
                target = view["by_depth"][str(depth)][family]
                assert target["raw_signed_shapley_by_layer"] == (
                    source["raw_signed_shapley_by_layer"]
                )
                assert target["positive_allocation_target_by_layer"] == (
                    source["positive_allocation_target_by_layer"]
                )
                assert set(target["standardized_signed_shapley_by_layer"]) == {
                    str(layer) for layer in players
                }
                for layer in players:
                    expected = subject.standardize_signed_shapley(
                        source["raw_signed_shapley_by_layer"][str(layer)],
                        normalization,
                        target_family=family,
                        prediction_depth=depth,
                        candidate_layer_id=layer,
                    )
                    assert target["standardized_signed_shapley_by_layer"][
                        str(layer)
                    ] == pytest.approx(expected, abs=1e-15)


def test_access_views_never_mutate_the_source_records(
    target_records: Any, normalization: Any
) -> None:
    training = fixtures.records_by_membership(target_records, "training")
    before = copy.deepcopy(list(training))
    subject.load_targets_for_access(
        training, access_mode="training_only", normalization=normalization
    )
    assert list(training) == before


# ---------------------------------------------------------------------------
# Story 3 — configuration, dry-run orchestration, and atomic persistence
# ---------------------------------------------------------------------------
#
# Contract assumed by the Story 3 tests below:
#
# * ``load_contribution_targets_config`` pins every scientific value of the
#   tracked Gate-C configuration and fails closed on drift. The tracked
#   configuration keeps ``official_materialization_enabled == false`` and only
#   ever accepts ``production`` input artifacts.
#
# * ``run_contribution_target_collection`` is the single shared plan-construction
#   path: source-training-only GT calibration, all 32 dual-family records, the
#   training-only Shapley normalization, all seven layered identities, and the
#   plan hash. It performs no I/O whatsoever.
#
# * ``dry_run_contribution_targets`` returns the complete plan with
#   ``artifact_written`` / ``run_directory_created`` false and
#   ``teacher_forward_count == 0``, and creates no file, directory, lock, or
#   temporary artifact.
#
# * Persistence is atomic and dual-hash: the ``.pt`` payload carries the
#   scientific record plus its embedded scientific hash, the file byte hash lives
#   only in the final manifest, and every write is followed by reload, rehash,
#   and revalidation before it may be marked verified. Runs are fresh-run only.


@pytest.fixture(scope="module")
def tracked_config() -> Any:
    return subject.load_contribution_targets_config(fixtures.TRACKED_CONFIG_PATH)


@pytest.fixture(scope="module")
def input_bundle(target_fixture: Any) -> Any:
    return fixtures.fixture_input_bundle(target_fixture)


@pytest.fixture(scope="module")
def collection(tracked_config: Any, input_bundle: Any) -> Any:
    return subject.run_contribution_target_collection(
        config=tracked_config, inputs=input_bundle
    )


def _snapshot_tree(root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    if not root.exists():
        return snapshot
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            snapshot[relative] = f"symlink:{os.readlink(path)}"
        elif path.is_dir():
            snapshot[relative] = "dir"
        else:
            snapshot[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def _official_config(tmp_path: Path, **overrides: Any) -> Any:
    return subject.load_contribution_targets_config(
        fixtures.write_controlled_official_config(tmp_path, **overrides)
    )


def _clone_contract_repository(tmp_path: Path, *, name: str = "contract-repository") -> Path:
    repository = tmp_path / name
    subprocess.run(
        ["git", "clone", "--quiet", "--no-local", str(fixtures.REPO_ROOT), str(repository)],
        check=True,
    )
    return repository


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=b2 negative control",
            "-c",
            "user.email=b2@example.invalid",
            *arguments,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _materialize(
    tmp_path: Path,
    config: Any,
    inputs: Any,
    *,
    run_name: str = "run",
    expected_plan_sha256: str | None = None,
) -> Any:
    run_dir = tmp_path / run_name
    if expected_plan_sha256 is None:
        expected_plan_sha256 = subject.run_contribution_target_collection(
            config=config, inputs=inputs
        ).plan["contribution_plan_scientific_sha256"]
    return subject.materialize_contribution_target_collection(
        config=config,
        inputs=inputs,
        output_run_dir=run_dir,
        expected_plan_sha256=expected_plan_sha256,
    )


# --- configuration ----------------------------------------------------------


def test_tracked_gate_c_config_loads_with_official_materialization_disabled(
    tracked_config: Any,
) -> None:
    assert tracked_config.configuration_id == "b2_contribution_targets_gate_c"
    assert tracked_config.contract_stage == "b2_04a"
    assert tracked_config.official_materialization_enabled is False
    assert tracked_config.expected_input_artifact_kind == subject.PRODUCTION_ARTIFACT_KIND
    assert tracked_config.candidate_layers == (6, 12, 18, 24)
    assert tracked_config.prediction_depths == (12, 18, 24)
    assert tracked_config.target_families == subject.TARGET_FAMILIES
    assert dict(tracked_config.split_counts) == {
        "training": 16,
        "calibration": 8,
        "evaluation": 8,
    }
    assert tracked_config.primary_target_dtype == "float64"
    assert tracked_config.resume_enabled is False
    assert tracked_config.dry_run_complete_compute is True
    assert tracked_config.expected_plan_sha_required_for_official is True


def test_official_b2_04b_config_has_independent_pinned_identity() -> None:
    official = subject.load_contribution_targets_config(fixtures.OFFICIAL_CONFIG_PATH)
    gate_c = subject.load_contribution_targets_config(fixtures.TRACKED_CONFIG_PATH)

    assert official.configuration_id == "b2_contribution_targets_official_v1"
    assert official.contract_stage == "b2_04b"
    assert official.official_materialization_enabled is True
    assert official.repository_identity_gate_enabled is True
    assert official.resume_enabled is False
    assert official.expected_plan_sha_required_for_official is True
    assert official.expected_contribution_contract_tag == "b2-contribution-target-contract-v1"
    assert official.expected_contribution_contract_commit == (
        "29591668c3228f6cebd7fd923ae1c39c6dad49bc"
    )

    identity_fields = {
        "configuration_id",
        "contract_stage",
        "official_materialization_enabled",
        "repository_identity_gate_enabled",
        "expected_contribution_contract_tag",
        "expected_contribution_contract_commit",
    }
    for field in official.__dataclass_fields__:
        if field not in identity_fields:
            assert getattr(official, field) == getattr(gate_c, field)


@pytest.mark.parametrize(
    "override",
    [
        {"contract_stage": "b2_04a"},
        {"official_materialization_enabled": False},
        {"repository_identity_gate_enabled": False},
        {"resume_enabled": True},
        {"expected_contribution_contract_tag": None},
        {"expected_contribution_contract_tag": "moved-tag"},
        {"expected_contribution_contract_commit": None},
        {"expected_contribution_contract_commit": "0" * 40},
    ],
)
def test_official_b2_04b_config_drift_fails_closed(
    tmp_path: Path, override: dict[str, Any]
) -> None:
    payload = fixtures.official_config_payload()
    payload.update(override)
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.load_contribution_targets_config(
            fixtures.write_config(tmp_path, payload, name="official-drift.json")
        )
    assert _error_code(excinfo) == "B2_CONTRIBUTION_CONFIG_DRIFT"


def test_tracked_config_carries_no_machine_local_paths() -> None:
    raw = fixtures.TRACKED_CONFIG_PATH.read_text(encoding="utf-8")
    for forbidden in ("/root", "/home", "/mnt", "autodl", "C:\\", "visa", "VisA"):
        assert forbidden not in raw


def test_config_missing_and_malformed_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.load_contribution_targets_config(tmp_path / "absent.json")
    assert _error_code(excinfo) == "B2_CONTRIBUTION_CONFIG_MISSING"
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.load_contribution_targets_config(broken)
    assert _error_code(excinfo) == "B2_CONTRIBUTION_CONFIG_INVALID"


@pytest.mark.parametrize(
    "overrides",
    [
        {"candidate_layers": [6, 12, 18]},
        {"prediction_depths": [12, 24]},
        {"gt_calibration": {"quantile_low": 0.05, "quantile_high": 0.995}},
        {"abnormal_gt_weights": {"pixel_ap": 0.5, "soft_dice": 0.4, "background_penalty": 0.1}},
        {"soft_dice_epsilon": 1e-3},
        {"split_counts": {"training": 24, "calibration": 4, "evaluation": 4}},
        {"primary_target_dtype": "float32"},
        {"coalition_fusion": "sum_preserving"},
        {"target_families": ["gt_localization"]},
        {"resume_enabled": True},
        {"expected_split_scientific_sha256": "0" * 64},
    ],
)
def test_config_drift_fails_closed(tmp_path: Path, overrides: dict[str, Any]) -> None:
    path = fixtures.write_config(
        tmp_path, {**fixtures.tracked_config_payload(), **overrides}, name="drift.json"
    )
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.load_contribution_targets_config(path)
    assert _error_code(excinfo) == "B2_CONTRIBUTION_CONFIG_DRIFT"


def test_tracked_configuration_id_may_never_enable_official_materialization(
    tmp_path: Path,
) -> None:
    payload = fixtures.tracked_config_payload()
    payload["official_materialization_enabled"] = True
    path = fixtures.write_config(tmp_path, payload, name="tracked_official.json")
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.load_contribution_targets_config(path)
    assert _error_code(excinfo) == "B2_CONTRIBUTION_CONFIG_DRIFT"


def test_tracked_configuration_id_may_never_accept_fixture_inputs(tmp_path: Path) -> None:
    payload = fixtures.tracked_config_payload()
    payload["expected_input_artifact_kind"] = "test_fixture"
    path = fixtures.write_config(tmp_path, payload, name="tracked_fixture.json")
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.load_contribution_targets_config(path)
    assert _error_code(excinfo) == "B2_CONTRIBUTION_CONFIG_DRIFT"


# --- shared collection path -------------------------------------------------


def test_collection_reproduces_the_independent_story_two_identities(
    collection: Any,
    target_fixture: Any,
    calibration_artifact: Any,
    target_records: Any,
    normalization: Any,
) -> None:
    assert collection.calibration_artifact["gt_map_calibration_scientific_sha256"] == (
        calibration_artifact["gt_map_calibration_scientific_sha256"]
    )
    assert [row["contribution_target_record_scientific_sha256"] for row in collection.records] == [
        row["contribution_target_record_scientific_sha256"]
        for row in sorted(target_records, key=lambda item: str(item["stable_sample_id"]))
    ]
    assert collection.normalization["shapley_normalization_scientific_sha256"] == (
        normalization["shapley_normalization_scientific_sha256"]
    )
    expected_plan = subject.build_contribution_plan(
        records=target_records,
        calibration_artifact=calibration_artifact,
        normalization=normalization,
        candidate_layers=target_fixture.candidate_layers,
        prediction_depths=target_fixture.prediction_depths,
        official_materialization_enabled=False,
    )
    assert collection.plan["contribution_plan_scientific_sha256"] == (
        expected_plan["contribution_plan_scientific_sha256"]
    )


def test_collection_uses_training_only_for_calibration_and_normalization(
    collection: Any, target_fixture: Any
) -> None:
    training_ids = sorted(
        sample.stable_sample_id for sample in target_fixture.by_membership("training")
    )
    assert list(
        collection.calibration_artifact["ordered_training_stable_sample_ids"]
    ) == training_ids
    assert list(
        collection.normalization["ordered_training_stable_sample_ids"]
    ) == training_ids


def test_collection_rejects_incomplete_split_coverage(
    tracked_config: Any, target_fixture: Any
) -> None:
    truncated = fixtures.fixture_input_bundle(
        target_fixture, samples=target_fixture.samples[:31]
    )
    with pytest.raises(subject.ContributionTargetError):
        subject.run_contribution_target_collection(config=tracked_config, inputs=truncated)


# --- dry run ----------------------------------------------------------------


def test_dry_run_returns_the_frozen_plan_semantics(
    tracked_config: Any, input_bundle: Any, collection: Any
) -> None:
    result = subject.dry_run_contribution_targets(
        config=tracked_config, inputs=input_bundle
    )
    assert result["mode"] == "dry_run"
    assert result["status"] == "passed"
    assert result["artifact_written"] is False
    assert result["run_directory_created"] is False
    assert result["teacher_forward_count"] == 0
    assert result["planned_samples"] == 32
    assert result["training_targets"] == 16
    assert result["calibration_targets"] == 8
    assert result["evaluation_targets"] == 8
    assert result["training_samples_for_gt_calibration"] == 16
    assert result["calibration_samples_for_gt_calibration"] == 0
    assert result["evaluation_samples_for_gt_calibration"] == 0
    assert result["training_samples_for_shapley_normalization"] == 16
    assert result["calibration_samples_for_shapley_normalization"] == 0
    assert result["evaluation_samples_for_shapley_normalization"] == 0
    assert list(result["prediction_depths"]) == [12, 18, 24]
    assert dict(result["coalition_counts"]) == {12: 4, 18: 8, 24: 16}
    assert result["contribution_plan_scientific_sha256"] == (
        collection.plan["contribution_plan_scientific_sha256"]
    )
    assert result["official_materialization_enabled"] is False


def test_dry_run_exposes_all_seven_layered_identities(
    tracked_config: Any, input_bundle: Any, collection: Any
) -> None:
    result = subject.dry_run_contribution_targets(
        config=tracked_config, inputs=input_bundle
    )
    for identity in subject.SEVEN_LAYERED_IDENTITY_KEYS:
        assert result[identity] == collection.plan[identity]
    assert len(subject.SEVEN_LAYERED_IDENTITY_KEYS) == 7
    assert set(subject.SEVEN_LAYERED_IDENTITY_KEYS) == {
        "gt_map_calibration_scientific_sha256",
        "contribution_target_sample_coverage_sha256",
        "contribution_target_collection_scientific_sha256",
        "shapley_normalization_scientific_sha256",
        "training_target_coverage_sha256",
        "calibration_target_coverage_sha256",
        "evaluation_target_coverage_sha256",
    }


def test_dry_run_creates_no_file_directory_lock_or_temporary(
    tracked_config: Any, input_bundle: Any, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    before = _snapshot_tree(workspace)
    subject.dry_run_contribution_targets(
        config=tracked_config,
        inputs=input_bundle,
        seed=0,
        output_dir=workspace / "never_created",
    )
    assert _snapshot_tree(workspace) == before
    assert not (workspace / "never_created").exists()


def test_dry_run_from_fixture_roots_matches_the_in_memory_dry_run(
    tracked_config: Any, target_fixture: Any, input_bundle: Any, tmp_path: Path
) -> None:
    layout = fixtures.prepare_hermetic_contribution_inputs(tmp_path, fixture=target_fixture)
    before = _snapshot_tree(tmp_path)
    result = subject.dry_run_contribution_targets_from_fixture_roots(
        config=tracked_config,
        teacher_cache_manifest_path=layout.teacher_cache_manifest,
        teacher_cache_root=layout.teacher_cache_root,
        descriptor_manifest_path=layout.descriptor_manifest,
        descriptor_root=layout.descriptor_root,
    )
    expected = subject.dry_run_contribution_targets(
        config=tracked_config, inputs=input_bundle
    )
    assert result["contribution_plan_scientific_sha256"] == (
        expected["contribution_plan_scientific_sha256"]
    )
    for identity in subject.SEVEN_LAYERED_IDENTITY_KEYS:
        assert result[identity] == expected[identity]
    assert result["artifact_written"] is False
    assert _snapshot_tree(tmp_path) == before


def test_fixture_inputs_are_refused_when_the_config_expects_production(
    tmp_path: Path, input_bundle: Any
) -> None:
    config = _official_config(
        tmp_path, expected_input_artifact_kind=subject.PRODUCTION_ARTIFACT_KIND
    )
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        _materialize(tmp_path, config, input_bundle)
    assert _error_code(excinfo) == "B2_TARGET_TEST_FIXTURE_NOT_ACCEPTED"
    assert not (tmp_path / "run").exists()


# --- historical accepted production manifests -------------------------------


LEGACY_TEACHER_CONFIGURATION_ID = "b2_teacher_cache_gate_c"
LEGACY_DESCRIPTOR_CONFIGURATION_ID = "b2_descriptor_artifacts_gate_c"


def _rewrite_as_historical_manifest(
    manifest_path: Path,
    *,
    configuration_id: str | None,
    artifact_kind: str | None = None,
) -> None:
    """Rewrite one manifest in the historical accepted schema.

    The accepted teacher-cache and descriptor manifests predate the
    ``artifact_kind`` field and identify themselves through ``configuration_id``.
    """

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.pop("artifact_kind", None)
    payload.pop("configuration_id", None)
    if configuration_id is not None:
        payload["configuration_id"] = configuration_id
    if artifact_kind is not None:
        payload["artifact_kind"] = artifact_kind
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _historical_manifest_target(layout: Any, label: str) -> tuple[Path, Path]:
    if label == "teacher-cache":
        return layout.teacher_cache_manifest, layout.teacher_cache_root
    return layout.descriptor_manifest, layout.descriptor_root


def test_loader_normalizes_historical_production_manifests_without_artifact_kind(
    tmp_path: Path,
) -> None:
    layout = fixtures.prepare_hermetic_contribution_inputs(tmp_path)
    for label, configuration_id in (
        ("teacher-cache", LEGACY_TEACHER_CONFIGURATION_ID),
        ("descriptor", LEGACY_DESCRIPTOR_CONFIGURATION_ID),
    ):
        manifest_path, _root = _historical_manifest_target(layout, label)
        _rewrite_as_historical_manifest(manifest_path, configuration_id=configuration_id)

    for label in ("teacher-cache", "descriptor"):
        manifest_path, root = _historical_manifest_target(layout, label)
        loaded = subject._load_input_manifest(
            manifest_path=manifest_path, root=root, label=label
        )
        assert loaded["artifact_kind"] == subject.PRODUCTION_ARTIFACT_KIND
        on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "artifact_kind" not in on_disk


@pytest.mark.parametrize(
    ("label", "configuration_id"),
    [
        ("teacher-cache", None),
        ("teacher-cache", "b2_teacher_cache_gate_b"),
        ("teacher-cache", LEGACY_DESCRIPTOR_CONFIGURATION_ID),
        ("descriptor", None),
        ("descriptor", "b2_descriptor_artifacts_gate_b"),
        ("descriptor", LEGACY_TEACHER_CONFIGURATION_ID),
    ],
)
def test_missing_artifact_kind_requires_the_exact_accepted_configuration_id(
    tmp_path: Path, label: str, configuration_id: str | None
) -> None:
    layout = fixtures.prepare_hermetic_contribution_inputs(tmp_path)
    manifest_path, root = _historical_manifest_target(layout, label)
    _rewrite_as_historical_manifest(manifest_path, configuration_id=configuration_id)
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject._load_input_manifest(manifest_path=manifest_path, root=root, label=label)
    assert _error_code(excinfo) == "B2_TARGET_ARTIFACT_KIND_INVALID"


@pytest.mark.parametrize("label", ["teacher-cache", "descriptor"])
@pytest.mark.parametrize(
    ("declared_kind", "accepted_kind"),
    [
        ("production", "production"),
        ("test_fixture", "test_fixture"),
        ("legacy_production", None),
        ("", None),
    ],
)
def test_declared_artifact_kind_is_never_inferred_away(
    tmp_path: Path, label: str, declared_kind: str, accepted_kind: str | None
) -> None:
    layout = fixtures.prepare_hermetic_contribution_inputs(tmp_path)
    manifest_path, root = _historical_manifest_target(layout, label)
    accepted_configuration_id = (
        LEGACY_TEACHER_CONFIGURATION_ID
        if label == "teacher-cache"
        else LEGACY_DESCRIPTOR_CONFIGURATION_ID
    )
    _rewrite_as_historical_manifest(
        manifest_path,
        configuration_id=accepted_configuration_id,
        artifact_kind=declared_kind,
    )
    if accepted_kind is None:
        with pytest.raises(subject.ContributionTargetError) as excinfo:
            subject._load_input_manifest(
                manifest_path=manifest_path, root=root, label=label
            )
        assert _error_code(excinfo) == "B2_TARGET_ARTIFACT_KIND_INVALID"
        return
    loaded = subject._load_input_manifest(
        manifest_path=manifest_path, root=root, label=label
    )
    assert loaded["artifact_kind"] == accepted_kind


def test_historical_production_manifests_still_face_every_downstream_pin(
    tmp_path: Path,
) -> None:
    layout = fixtures.prepare_hermetic_contribution_inputs(tmp_path)
    _rewrite_as_historical_manifest(
        layout.teacher_cache_manifest, configuration_id=LEGACY_TEACHER_CONFIGURATION_ID
    )
    _rewrite_as_historical_manifest(
        layout.descriptor_manifest, configuration_id=LEGACY_DESCRIPTOR_CONFIGURATION_ID
    )
    config = subject.load_contribution_targets_config(fixtures.TRACKED_CONFIG_PATH)
    assert config.expected_input_artifact_kind == subject.PRODUCTION_ARTIFACT_KIND
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.load_contribution_inputs_from_disk(
            config=config,
            teacher_cache_manifest_path=layout.teacher_cache_manifest,
            teacher_cache_root=layout.teacher_cache_root,
            descriptor_manifest_path=layout.descriptor_manifest,
            descriptor_root=layout.descriptor_root,
            mvtec_root=layout.mvtec_root,
        )
    assert _error_code(excinfo) == "B2_CONTRIBUTION_UPSTREAM_IDENTITY_MISMATCH"


def test_historical_manifest_pair_must_agree_on_the_inferred_kind(
    tmp_path: Path,
) -> None:
    layout = fixtures.prepare_hermetic_contribution_inputs(tmp_path)
    _rewrite_as_historical_manifest(
        layout.teacher_cache_manifest, configuration_id=LEGACY_TEACHER_CONFIGURATION_ID
    )
    config = subject.load_contribution_targets_config(fixtures.TRACKED_CONFIG_PATH)
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.load_contribution_inputs_from_disk(
            config=config,
            teacher_cache_manifest_path=layout.teacher_cache_manifest,
            teacher_cache_root=layout.teacher_cache_root,
            descriptor_manifest_path=layout.descriptor_manifest,
            descriptor_root=layout.descriptor_root,
            mvtec_root=layout.mvtec_root,
        )
    assert _error_code(excinfo) == "B2_TARGET_ARTIFACT_KIND_INVALID"


def test_fixture_loader_accepts_mvtec_root_without_using_it(
    tmp_path: Path, tracked_config: Any, target_fixture: Any
) -> None:
    layout = fixtures.prepare_hermetic_contribution_inputs(tmp_path, fixture=target_fixture)
    bundle = subject.load_contribution_inputs_from_disk(
        config=tracked_config,
        teacher_cache_manifest_path=layout.teacher_cache_manifest,
        teacher_cache_root=layout.teacher_cache_root,
        descriptor_manifest_path=layout.descriptor_manifest,
        descriptor_root=layout.descriptor_root,
        mvtec_root=layout.mvtec_root,
    )
    assert bundle.artifact_kind == subject.TEST_FIXTURE_ARTIFACT_KIND
    assert len(bundle.samples) == len(target_fixture.samples)


def test_production_inputs_require_an_explicit_mvtec_root(tmp_path: Path) -> None:
    layout = fixtures.prepare_hermetic_contribution_inputs(tmp_path)
    _rewrite_as_historical_manifest(
        layout.teacher_cache_manifest, configuration_id=LEGACY_TEACHER_CONFIGURATION_ID
    )
    _rewrite_as_historical_manifest(
        layout.descriptor_manifest, configuration_id=LEGACY_DESCRIPTOR_CONFIGURATION_ID
    )
    config = subject.load_contribution_targets_config(fixtures.TRACKED_CONFIG_PATH)
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.load_contribution_inputs_from_disk(
            config=config,
            teacher_cache_manifest_path=layout.teacher_cache_manifest,
            teacher_cache_root=layout.teacher_cache_root,
            descriptor_manifest_path=layout.descriptor_manifest,
            descriptor_root=layout.descriptor_root,
        )
    assert _error_code(excinfo) == "B2_CONTRIBUTION_SOURCE_ROOT_REQUIRED"


def test_production_maps_and_mask_are_read_from_teacher_scientific_tensors() -> None:
    layers = (6, 12)
    depths = (12,)
    shape = (1, 1, 4, 4)
    maps = {
        12: {
            6: torch.full(shape, 0.25, dtype=torch.float32),
            12: torch.full(shape, 0.75, dtype=torch.float32),
        }
    }
    full_depth = torch.full(shape, 0.5, dtype=torch.float32)
    mask = torch.zeros(shape, dtype=torch.float32)
    mask[..., 1:3, 1:3] = 1.0
    tensors = {
        "causal_map:12:6": {"tensor": maps[12][6]},
        "causal_map:12:12": {"tensor": maps[12][12]},
        "full_depth_map": {"tensor": full_depth},
        "anomalous_mask": {"tensor": mask},
    }
    teacher_record = {
        "candidate_layers": list(layers),
        "prediction_depths": list(depths),
        "image_label": 1,
        "tensors": tensors,
    }
    extracted_maps, extracted_full = subject.production_maps_from_teacher_record(
        teacher_record
    )
    assert set(extracted_maps) == {12}
    assert set(extracted_maps[12]) == {6, 12}
    assert torch.equal(extracted_maps[12][6], maps[12][6])
    assert torch.equal(extracted_full, full_depth)
    extracted_mask = subject.production_mask_from_teacher_record(teacher_record)
    assert torch.equal(extracted_mask, mask)

    normal_record = {
        "candidate_layers": list(layers),
        "prediction_depths": list(depths),
        "image_label": 0,
        "tensors": {
            "causal_map:12:6": {"tensor": maps[12][6]},
            "causal_map:12:12": {"tensor": maps[12][12]},
            "full_depth_map": {"tensor": full_depth},
        },
    }
    zero_mask = subject.production_mask_from_teacher_record(normal_record)
    assert tuple(zero_mask.shape) == shape
    assert float(zero_mask.sum()) == 0.0


def test_production_teacher_payload_rejects_missing_causal_map_tensor() -> None:
    teacher_record = {
        "candidate_layers": [6, 12],
        "prediction_depths": [12],
        "image_label": 0,
        "tensors": {"full_depth_map": {"tensor": torch.zeros((1, 1, 2, 2))}},
    }
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.production_maps_from_teacher_record(teacher_record)
    assert _error_code(excinfo) == "B2_CONTRIBUTION_TEACHER_PAYLOAD_SCHEMA_INVALID"


# --- official materialization gate -----------------------------------------


def test_repository_identity_verifier_accepts_a_clean_descendant(tmp_path: Path) -> None:
    repository = _clone_contract_repository(tmp_path)
    config = subject.load_contribution_targets_config(fixtures.OFFICIAL_CONFIG_PATH)
    identity = subject.verify_contribution_repository_identity(
        config=config, repository_root=repository
    )
    assert identity.contract_tag == "b2-contribution-target-contract-v1"
    assert identity.contract_commit == "29591668c3228f6cebd7fd923ae1c39c6dad49bc"
    assert identity.generation_commit == subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert identity.head_is_descendant is True
    assert identity.worktree_clean is True


def test_repository_identity_verifier_rejects_missing_or_moved_tag(tmp_path: Path) -> None:
    repository = _clone_contract_repository(tmp_path)
    config = subject.load_contribution_targets_config(fixtures.OFFICIAL_CONFIG_PATH)
    subprocess.run(
        ["git", "-C", str(repository), "tag", "-d", config.expected_contribution_contract_tag],
        check=True,
        capture_output=True,
    )
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.verify_contribution_repository_identity(
            config=config, repository_root=repository
        )
    assert _error_code(excinfo) == "B2_CONTRIBUTION_CONTRACT_TAG_INVALID"

    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "tag",
            config.expected_contribution_contract_tag,
            "HEAD",
        ],
        check=True,
    )
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.verify_contribution_repository_identity(
            config=config, repository_root=repository
        )
    assert _error_code(excinfo) == "B2_CONTRIBUTION_CONTRACT_TAG_INVALID"


def test_repository_identity_verifier_rejects_dirty_and_observed_head_mismatch(
    tmp_path: Path,
) -> None:
    repository = _clone_contract_repository(tmp_path)
    config = subject.load_contribution_targets_config(fixtures.OFFICIAL_CONFIG_PATH)
    (repository / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.verify_contribution_repository_identity(
            config=config, repository_root=repository
        )
    assert _error_code(excinfo) == "B2_CONTRIBUTION_WORKTREE_DIRTY"
    (repository / "untracked.txt").unlink()

    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.verify_contribution_repository_identity(
            config=config,
            repository_root=repository,
            expected_generation_commit="0" * 40,
        )
    assert _error_code(excinfo) == "B2_CONTRIBUTION_GENERATION_COMMIT_CHANGED"


def test_official_api_requires_repository_root_before_materialization(
    tmp_path: Path, input_bundle: Any
) -> None:
    controlled = _official_config(tmp_path)
    config = dataclasses.replace(
        controlled,
        repository_identity_gate_enabled=True,
        expected_contribution_contract_tag="b2-contribution-target-contract-v1",
        expected_contribution_contract_commit="29591668c3228f6cebd7fd923ae1c39c6dad49bc",
    )
    expected = subject.run_contribution_target_collection(
        config=config, inputs=input_bundle
    ).plan["contribution_plan_scientific_sha256"]
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.materialize_contribution_target_collection(
            config=config,
            inputs=input_bundle,
            output_run_dir=tmp_path / "run",
            expected_plan_sha256=expected,
        )
    assert _error_code(excinfo) == "B2_CONTRIBUTION_REPOSITORY_ROOT_REQUIRED"
    assert not (tmp_path / "run").exists()


def test_official_api_rechecks_repository_identity_before_any_write(
    tmp_path: Path, input_bundle: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _clone_contract_repository(tmp_path)
    controlled = _official_config(tmp_path)
    config = dataclasses.replace(
        controlled,
        repository_identity_gate_enabled=True,
        expected_contribution_contract_tag="b2-contribution-target-contract-v1",
        expected_contribution_contract_commit="29591668c3228f6cebd7fd923ae1c39c6dad49bc",
    )
    expected = subject.run_contribution_target_collection(
        config=config, inputs=input_bundle
    ).plan["contribution_plan_scientific_sha256"]
    production_collection = subject.run_contribution_target_collection

    def mutate_after_plan(**kwargs: Any) -> Any:
        collection = production_collection(**kwargs)
        (repository / "mutation-after-plan.txt").write_text("dirty\n", encoding="utf-8")
        return collection

    monkeypatch.setattr(
        subject, "run_contribution_target_collection", mutate_after_plan
    )
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.materialize_contribution_target_collection(
            config=config,
            inputs=input_bundle,
            output_run_dir=tmp_path / "run",
            expected_plan_sha256=expected,
            repository_root=repository,
        )
    assert _error_code(excinfo) == "B2_CONTRIBUTION_WORKTREE_DIRTY"
    assert not (tmp_path / "run").exists()


def test_non_dry_run_is_refused_while_official_materialization_is_disabled(
    tracked_config: Any, input_bundle: Any, tmp_path: Path
) -> None:
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.materialize_contribution_target_collection(
            config=tracked_config,
            inputs=input_bundle,
            output_run_dir=tmp_path / "run",
            expected_plan_sha256="0" * 64,
        )
    assert _error_code(excinfo) == "B2_CONTRIBUTION_OFFICIAL_MATERIALIZATION_NOT_ENABLED"
    assert not (tmp_path / "run").exists()


def test_official_materialization_requires_the_expected_plan_hash(
    tmp_path: Path, input_bundle: Any
) -> None:
    config = _official_config(tmp_path)
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.materialize_contribution_target_collection(
            config=config,
            inputs=input_bundle,
            output_run_dir=tmp_path / "run",
            expected_plan_sha256=None,
        )
    assert _error_code(excinfo) == "B2_CONTRIBUTION_EXPECTED_PLAN_SHA_MISSING"
    assert not (tmp_path / "run").exists()


def test_official_materialization_rejects_a_malformed_expected_plan_hash(
    tmp_path: Path, input_bundle: Any
) -> None:
    config = _official_config(tmp_path)
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.materialize_contribution_target_collection(
            config=config,
            inputs=input_bundle,
            output_run_dir=tmp_path / "run",
            expected_plan_sha256="not-a-sha",
        )
    assert _error_code(excinfo) == "B2_CONTRIBUTION_EXPECTED_PLAN_SHA_MALFORMED"
    assert not (tmp_path / "run").exists()


def test_expected_plan_hash_mismatch_fails_before_any_write(
    tmp_path: Path, input_bundle: Any
) -> None:
    config = _official_config(tmp_path)
    before = _snapshot_tree(tmp_path)
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.materialize_contribution_target_collection(
            config=config,
            inputs=input_bundle,
            output_run_dir=tmp_path / "run",
            expected_plan_sha256="a" * 64,
        )
    assert _error_code(excinfo) == "B2_CONTRIBUTION_EXPECTED_PLAN_SHA_MISMATCH"
    assert not (tmp_path / "run").exists()
    assert _snapshot_tree(tmp_path) == before


def test_controlled_official_materialization_produces_a_verified_collection(
    tmp_path: Path, input_bundle: Any
) -> None:
    config = _official_config(tmp_path)
    expected = subject.run_contribution_target_collection(
        config=config, inputs=input_bundle
    )
    result = _materialize(
        tmp_path,
        config,
        input_bundle,
        expected_plan_sha256=expected.plan["contribution_plan_scientific_sha256"],
    )
    run_dir = tmp_path / "run"
    assert result.run_dir == run_dir
    assert result.teacher_forward_count == 0
    assert sorted(path.name for path in (run_dir / "records").glob("*.pt")) == sorted(
        f"{row['stable_sample_id']}.pt" for row in expected.records
    )
    assert (run_dir / subject.CALIBRATION_RELATIVE_PATH).is_file()
    assert (run_dir / subject.NORMALIZATION_RELATIVE_PATH).is_file()
    assert (run_dir / "final_manifest.json").is_file()
    assert (run_dir / "final_manifest.json.sha256").is_file()
    assert result.manifest["status"] == "passed"
    assert result.manifest["contribution_plan_scientific_sha256"] == (
        expected.plan["contribution_plan_scientific_sha256"]
    )
    assert result.manifest["artifact_kind"] == "test_fixture"
    verified = subject.verify_contribution_target_collection(config=config, run_dir=run_dir)
    assert len(verified.records_by_id) == 32
    assert verified.teacher_forward_count == 0
    assert not list(run_dir.rglob("*.tmp"))


# --- persistence ------------------------------------------------------------


def test_record_payload_is_dual_hash_and_excludes_the_file_hash(
    tmp_path: Path, tracked_config: Any, collection: Any
) -> None:
    record = dict(collection.records[0])
    destination = tmp_path / subject.contribution_record_relative_path(
        record["stable_sample_id"]
    )
    entry = subject.write_contribution_target_record_atomic(
        destination,
        record,
        candidate_layers=tracked_config.candidate_layers,
        prediction_depths=tracked_config.prediction_depths,
    )
    assert entry.verification_status == "verified"
    payload = torch.load(destination, map_location="cpu", weights_only=True)
    assert set(payload) == {
        "scientific_record",
        "contribution_target_record_scientific_sha256",
    }
    assert "contribution_target_record_file_sha256" not in payload["scientific_record"]
    assert payload["contribution_target_record_scientific_sha256"] == (
        record["contribution_target_record_scientific_sha256"]
    )
    assert entry.contribution_target_record_file_sha256 == hashlib.sha256(
        destination.read_bytes()
    ).hexdigest()
    assert not list(tmp_path.rglob("*.tmp"))


def test_record_write_refuses_to_overwrite_an_existing_artifact(
    tmp_path: Path, tracked_config: Any, collection: Any
) -> None:
    record = dict(collection.records[0])
    destination = tmp_path / subject.contribution_record_relative_path(
        record["stable_sample_id"]
    )
    subject.write_contribution_target_record_atomic(
        destination,
        record,
        candidate_layers=tracked_config.candidate_layers,
        prediction_depths=tracked_config.prediction_depths,
    )
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.write_contribution_target_record_atomic(
            destination,
            record,
            candidate_layers=tracked_config.candidate_layers,
            prediction_depths=tracked_config.prediction_depths,
        )
    assert _error_code(excinfo) == "B2_CONTRIBUTION_OVERWRITE_FORBIDDEN"


def test_record_write_rejects_a_scientific_hash_that_does_not_match_content(
    tmp_path: Path, tracked_config: Any, collection: Any
) -> None:
    record = copy.deepcopy(dict(collection.records[0]))
    record["category"] = "tampered_category"
    destination = tmp_path / subject.contribution_record_relative_path(
        record["stable_sample_id"]
    )
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.write_contribution_target_record_atomic(
            destination,
            record,
            candidate_layers=tracked_config.candidate_layers,
            prediction_depths=tracked_config.prediction_depths,
        )
    assert _error_code(excinfo) == "B2_TARGET_RECORD_HASH_MISMATCH"
    assert not destination.exists()
    assert not list(tmp_path.rglob("*.tmp"))


def test_output_collision_and_resume_fail_closed(
    tmp_path: Path, input_bundle: Any
) -> None:
    config = _official_config(tmp_path)
    _materialize(tmp_path, config, input_bundle)
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        _materialize(tmp_path, config, input_bundle)
    assert _error_code(excinfo) == "B2_CONTRIBUTION_OUTPUT_DIR_EXISTS"

    partial = tmp_path / "partial"
    (partial / "records").mkdir(parents=True)
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.materialize_contribution_target_collection(
            config=config,
            inputs=input_bundle,
            output_run_dir=partial,
            expected_plan_sha256=subject.run_contribution_target_collection(
                config=config, inputs=input_bundle
            ).plan["contribution_plan_scientific_sha256"],
        )
    assert _error_code(excinfo) == "B2_CONTRIBUTION_OUTPUT_DIR_EXISTS"
    assert config.resume_enabled is False


def test_digest_only_record_payload_is_rejected_on_verification(
    tmp_path: Path, input_bundle: Any
) -> None:
    config = _official_config(tmp_path)
    result = _materialize(tmp_path, config, input_bundle)
    run_dir = result.run_dir
    stable_id = str(result.manifest["records"][0]["stable_sample_id"])
    victim = run_dir / subject.contribution_record_relative_path(stable_id)
    payload = torch.load(victim, map_location="cpu", weights_only=True)
    stripped = dict(payload["scientific_record"])
    stripped.pop("depth_targets")
    victim.unlink()
    torch.save(
        {
            "scientific_record": stripped,
            "contribution_target_record_scientific_sha256": payload[
                "contribution_target_record_scientific_sha256"
            ],
        },
        victim,
    )
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.verify_contribution_target_collection(config=config, run_dir=run_dir)
    assert _error_code(excinfo) in {
        "B2_CONTRIBUTION_DIGEST_ONLY_RECORD",
        "B2_CONTRIBUTION_RECORD_FILE_HASH_MISMATCH",
    }


def test_record_file_hash_mismatch_is_detected(tmp_path: Path, input_bundle: Any) -> None:
    config = _official_config(tmp_path)
    result = _materialize(tmp_path, config, input_bundle)
    run_dir = result.run_dir
    stable_id = str(result.manifest["records"][0]["stable_sample_id"])
    victim = run_dir / subject.contribution_record_relative_path(stable_id)
    victim.write_bytes(victim.read_bytes() + b"\x00")
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.verify_contribution_target_collection(config=config, run_dir=run_dir)
    assert _error_code(excinfo) == "B2_CONTRIBUTION_RECORD_FILE_HASH_MISMATCH"


def test_record_scientific_hash_mismatch_is_detected(
    tmp_path: Path, input_bundle: Any
) -> None:
    config = _official_config(tmp_path)
    result = _materialize(tmp_path, config, input_bundle)
    run_dir = result.run_dir
    stable_id = str(result.manifest["records"][0]["stable_sample_id"])
    entry = subject.PersistedTargetRecordEntry(
        stable_sample_id=stable_id,
        relative_record_path=subject.contribution_record_relative_path(stable_id),
        contribution_target_record_scientific_sha256="b" * 64,
        contribution_target_record_file_sha256=hashlib.sha256(
            (run_dir / subject.contribution_record_relative_path(stable_id)).read_bytes()
        ).hexdigest(),
    )
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.verify_persisted_contribution_target_record(
            run_dir=run_dir,
            entry=entry,
            candidate_layers=config.candidate_layers,
            prediction_depths=config.prediction_depths,
        )
    assert _error_code(excinfo) == "B2_TARGET_RECORD_HASH_MISMATCH"


def test_missing_shapley_normalization_statistics_fail_closed(
    tmp_path: Path, input_bundle: Any
) -> None:
    config = _official_config(tmp_path)
    result = _materialize(tmp_path, config, input_bundle)
    (result.run_dir / subject.NORMALIZATION_RELATIVE_PATH).unlink()
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.verify_contribution_target_collection(config=config, run_dir=result.run_dir)
    assert _error_code(excinfo) == "B2_CONTRIBUTION_MISSING_ARTIFACT"


def test_tampered_shapley_normalization_statistics_fail_closed(
    tmp_path: Path, input_bundle: Any
) -> None:
    config = _official_config(tmp_path)
    result = _materialize(tmp_path, config, input_bundle)
    victim = result.run_dir / subject.NORMALIZATION_RELATIVE_PATH
    victim.write_bytes(victim.read_bytes() + b"\x00")
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.verify_contribution_target_collection(config=config, run_dir=result.run_dir)
    assert _error_code(excinfo) == "B2_CONTRIBUTION_NORMALIZATION_FILE_HASH_MISMATCH"


def test_tampered_gt_map_calibration_statistics_fail_closed(
    tmp_path: Path, input_bundle: Any
) -> None:
    config = _official_config(tmp_path)
    result = _materialize(tmp_path, config, input_bundle)
    victim = result.run_dir / subject.CALIBRATION_RELATIVE_PATH
    victim.write_bytes(victim.read_bytes() + b"\x00")
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.verify_contribution_target_collection(config=config, run_dir=result.run_dir)
    assert _error_code(excinfo) == "B2_CONTRIBUTION_CALIBRATION_FILE_HASH_MISMATCH"


def test_missing_and_mismatched_final_manifest_receipt_fail_closed(
    tmp_path: Path, input_bundle: Any
) -> None:
    config = _official_config(tmp_path)
    result = _materialize(tmp_path, config, input_bundle)
    receipt = result.run_dir / "final_manifest.json.sha256"
    receipt.write_text("c" * 64 + "\n", encoding="utf-8")
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.verify_final_manifest_receipt(result.run_dir)
    assert _error_code(excinfo) == "B2_CONTRIBUTION_MANIFEST_RECEIPT_MISMATCH"
    receipt.unlink()
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.verify_final_manifest_receipt(result.run_dir)
    assert _error_code(excinfo) == "B2_CONTRIBUTION_MISSING_ARTIFACT"


def test_orphan_extra_and_temporary_artifacts_fail_the_integrity_audit(
    tmp_path: Path, input_bundle: Any
) -> None:
    config = _official_config(tmp_path)
    result = _materialize(tmp_path, config, input_bundle)
    run_dir = result.run_dir
    planned = [str(row["stable_sample_id"]) for row in result.manifest["records"]]
    subject.audit_contribution_target_integrity(
        run_dir=run_dir, manifest=result.manifest, planned_ids=planned
    )

    orphan = run_dir / "records" / f"{'f' * 64}.pt"
    orphan.write_bytes(b"orphan")
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.audit_contribution_target_integrity(
            run_dir=run_dir, manifest=result.manifest, planned_ids=planned
        )
    assert _error_code(excinfo) == "B2_CONTRIBUTION_ORPHAN_ARTIFACT"
    orphan.unlink()

    temporary = run_dir / "records" / ".leftover.pt.tmp"
    temporary.write_bytes(b"temp")
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.audit_contribution_target_integrity(
            run_dir=run_dir, manifest=result.manifest, planned_ids=planned
        )
    assert _error_code(excinfo) == "B2_CONTRIBUTION_TEMP_ARTIFACT_PRESENT"
    temporary.unlink()

    missing = run_dir / subject.contribution_record_relative_path(planned[0])
    missing.unlink()
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.audit_contribution_target_integrity(
            run_dir=run_dir, manifest=result.manifest, planned_ids=planned
        )
    assert _error_code(excinfo) == "B2_CONTRIBUTION_PARTIAL_CLAIMING_PASSED"


@pytest.mark.parametrize(
    "relative",
    ["../escape.pt", "records/../../escape.pt", "/absolute/escape.pt", "records/./"],
)
def test_run_relative_paths_reject_traversal(
    tmp_path: Path, input_bundle: Any, relative: str
) -> None:
    config = _official_config(tmp_path)
    result = _materialize(tmp_path, config, input_bundle)
    stable_id = str(result.manifest["records"][0]["stable_sample_id"])
    entry = subject.PersistedTargetRecordEntry(
        stable_sample_id=stable_id,
        relative_record_path=relative,
        contribution_target_record_scientific_sha256="d" * 64,
        contribution_target_record_file_sha256="e" * 64,
    )
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.verify_persisted_contribution_target_record(
            run_dir=result.run_dir,
            entry=entry,
            candidate_layers=config.candidate_layers,
            prediction_depths=config.prediction_depths,
        )
    assert _error_code(excinfo) == "B2_CONTRIBUTION_RUN_RELATIVE_PATH_INVALID"


def test_symlink_escape_from_the_run_directory_is_rejected(
    tmp_path: Path, input_bundle: Any
) -> None:
    config = _official_config(tmp_path)
    result = _materialize(tmp_path, config, input_bundle)
    run_dir = result.run_dir
    outside = tmp_path / "outside.pt"
    stable_id = str(result.manifest["records"][0]["stable_sample_id"])
    victim = run_dir / subject.contribution_record_relative_path(stable_id)
    outside.write_bytes(victim.read_bytes())
    victim.unlink()
    victim.symlink_to(outside)
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.verify_contribution_target_collection(config=config, run_dir=run_dir)
    assert _error_code(excinfo) == "B2_CONTRIBUTION_RUN_ROOT_ESCAPE"


def test_passed_manifest_requires_verified_record_entries(
    tmp_path: Path, tracked_config: Any, collection: Any
) -> None:
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.build_contribution_targets_manifest(
            config=tracked_config,
            collection=collection,
            record_entries=(),
            calibration_entry=None,
            normalization_entry=None,
        )
    assert _error_code(excinfo) == (
        "B2_CONTRIBUTION_PASSED_MANIFEST_REQUIRES_VERIFIED_RECORDS"
    )


# --- verified collection comparison ----------------------------------------


def test_comparison_accepts_two_independently_verified_identical_runs(
    tmp_path: Path, input_bundle: Any
) -> None:
    config = _official_config(tmp_path)
    first_result = _materialize(tmp_path, config, input_bundle, run_name="run-a")
    second_result = _materialize(tmp_path, config, input_bundle, run_name="run-b")
    first = subject.verify_contribution_target_collection(
        config=config, run_dir=first_result.run_dir
    )
    second = subject.verify_contribution_target_collection(
        config=config, run_dir=second_result.run_dir
    )
    comparison = subject.compare_contribution_target_collections(
        first=first, second=second
    )
    assert comparison.scientifically_equivalent is True
    assert comparison.reasons == ()
    for field in dataclasses.fields(comparison):
        if field.name not in {"scientifically_equivalent", "reasons"}:
            assert getattr(comparison, field.name) is True


def test_comparison_rejects_unverified_inputs(
    tmp_path: Path, input_bundle: Any
) -> None:
    config = _official_config(tmp_path)
    result = _materialize(tmp_path, config, input_bundle)
    verified = subject.verify_contribution_target_collection(
        config=config, run_dir=result.run_dir
    )
    replaced = dataclasses.replace(verified)
    copied = subject.VerifiedContributionTargetCollection(
        **{
            field.name: getattr(verified, field.name)
            for field in dataclasses.fields(verified)
            if field.init
        }
    )
    for candidate in (result.manifest, result, replaced, copied):
        for pair in (
            {"first": candidate, "second": verified},
            {"first": verified, "second": candidate},
        ):
            with pytest.raises(subject.ContributionTargetError) as excinfo:
                subject.compare_contribution_target_collections(**pair)
            assert _error_code(excinfo) == "B2_CONTRIBUTION_COLLECTION_NOT_VERIFIED"


def test_verification_seal_is_instance_identity_not_a_copyable_token(
    tmp_path: Path, input_bundle: Any
) -> None:
    config = _official_config(tmp_path)
    result = _materialize(tmp_path, config, input_bundle)
    verified = subject.verify_contribution_target_collection(
        config=config, run_dir=result.run_dir
    )
    assert [
        field.name
        for field in dataclasses.fields(subject.VerifiedContributionTargetCollection)
        if field.name.startswith("_")
    ] == []
    assert subject.compare_contribution_target_collections(
        first=verified, second=verified
    ).scientifically_equivalent is True


def test_comparison_reports_descriptor_variant_scientific_mismatch(
    tmp_path: Path,
) -> None:
    config = _official_config(tmp_path)
    fixture_a = fixtures.build_contribution_target_fixture(descriptor_variant="A")
    fixture_b = fixtures.build_contribution_target_fixture(descriptor_variant="B")
    first_result = _materialize(
        tmp_path,
        config,
        fixtures.fixture_input_bundle(fixture_a),
        run_name="run-a",
    )
    second_result = _materialize(
        tmp_path,
        config,
        fixtures.fixture_input_bundle(fixture_b),
        run_name="run-b",
    )
    comparison = subject.compare_contribution_target_collections(
        first=subject.verify_contribution_target_collection(
            config=config, run_dir=first_result.run_dir
        ),
        second=subject.verify_contribution_target_collection(
            config=config, run_dir=second_result.run_dir
        ),
    )
    assert comparison.scientifically_equivalent is False
    assert comparison.layered_identities_equal is False
    assert comparison.record_scientific_hashes_equal is False
    assert any("descriptor" in reason for reason in comparison.reasons)


# --- categorized comparison mismatches --------------------------------------


COMPARISON_MISMATCH_CASES: dict[str, tuple[str, str]] = {
    "layered_identity_drift": (
        "layered_identities_equal",
        "layered scientific identity",
    ),
    "record_scientific_hash_drift": (
        "record_scientific_hashes_equal",
        "record scientific hash",
    ),
    "teacher_cache_identity_drift": (
        "record_scientific_hashes_equal",
        "teacher_cache_scientific_sha256",
    ),
    "descriptor_collection_identity_drift": (
        "record_scientific_hashes_equal",
        "descriptor_collection_scientific_sha256",
    ),
    "descriptor_record_identity_drift": (
        "record_scientific_hashes_equal",
        "descriptor_record_scientific_sha256",
    ),
    "coalition_utility_component_drift": (
        "utility_tables_equal",
        "coalition utility component",
    ),
    "raw_utility_drift": ("utility_tables_equal", "raw utility"),
    "centered_value_drift": ("utility_tables_equal", "centered value"),
    "empty_coalition_raw_utility_drift": (
        "utility_tables_equal",
        "empty coalition raw utility",
    ),
    "grand_coalition_centered_value_drift": (
        "utility_tables_equal",
        "grand coalition centered value",
    ),
    "signed_shapley_drift": ("signed_shapley_equal", "signed Shapley"),
    "efficiency_residual_drift": ("signed_shapley_equal", "efficiency residual"),
    "allocation_drift": ("allocations_equal", "allocation"),
    "changed_split_membership": ("coverage_equal", "split membership"),
    "gt_calibration_statistic_drift": ("gt_calibration_equal", "GT calibration"),
    "shapley_normalization_statistic_drift": (
        "shapley_normalization_equal",
        "Shapley normalization",
    ),
    "nonzero_teacher_forward_count": (
        "teacher_forward_count_equal",
        "teacher forward count",
    ),
}

_IDENTITY_DRIFT_FIELDS = {
    "teacher_cache_identity_drift": "teacher_cache_scientific_sha256",
    "descriptor_collection_identity_drift": "descriptor_collection_scientific_sha256",
    "descriptor_record_identity_drift": "descriptor_record_scientific_sha256",
}


def _mutate_comparison_payload(payload: dict[str, Any], case: str) -> None:
    delta = 1e-9
    stable_id = sorted(payload["records_by_id"])[0]
    record = payload["records_by_id"][stable_id]
    depth_key = sorted(record["depth_targets"])[0]
    depth = record["depth_targets"][depth_key]
    coalition = depth["coalition_table"][1]
    family = depth["gt_localization"]
    if case == "layered_identity_drift":
        payload["manifest"]["contribution_plan_scientific_sha256"] = "0" * 64
    elif case == "record_scientific_hash_drift":
        record["contribution_target_record_scientific_sha256"] = "0" * 64
    elif case in _IDENTITY_DRIFT_FIELDS:
        record[_IDENTITY_DRIFT_FIELDS[case]] = "0" * 64
    elif case == "coalition_utility_component_drift":
        components = coalition["gt_localization"]["utility_components"]
        component = sorted(components)[0]
        components[component] = float(components[component]) + delta
    elif case == "raw_utility_drift":
        coalition["gt_localization"]["raw_utility"] += delta
    elif case == "centered_value_drift":
        coalition["teacher_fidelity"]["centered_value"] += delta
    elif case == "empty_coalition_raw_utility_drift":
        family["empty_coalition_raw_utility"] += delta
    elif case == "grand_coalition_centered_value_drift":
        family["grand_coalition_centered_value"] += delta
    elif case == "signed_shapley_drift":
        layer = sorted(family["raw_signed_shapley_by_layer"])[0]
        family["raw_signed_shapley_by_layer"][layer] += delta
    elif case == "efficiency_residual_drift":
        family["efficiency_residual"] += delta
    elif case == "allocation_drift":
        allocation = family["positive_allocation_target_by_layer"]
        layers = sorted(allocation)
        allocation[layers[0]], allocation[layers[-1]] = (
            allocation[layers[-1]],
            allocation[layers[0]],
        )
    elif case == "changed_split_membership":
        record["split_membership"] = (
            "calibration" if record["split_membership"] == "training" else "training"
        )
    elif case == "gt_calibration_statistic_drift":
        by_depth = payload["calibration_artifact"]["by_depth"]
        payload["calibration_artifact"]["by_depth"][sorted(by_depth)[0]]["q_low"] += delta
    elif case == "shapley_normalization_statistic_drift":
        axes = payload["normalization"]["axes"]["gt_localization"]
        axes[sorted(axes)[0]]["layers"][0]["mean"] += delta
    elif case == "nonzero_teacher_forward_count":
        payload["teacher_forward_count"] = 1
    else:  # pragma: no cover - guards against silent placeholder cases
        raise AssertionError(f"unmapped categorized mismatch case: {case}")


@pytest.fixture(scope="module")
def verified_reference_payload(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("categorized-comparison")
    config = _official_config(root)
    fixture = fixtures.build_contribution_target_fixture()
    result = _materialize(root, config, fixtures.fixture_input_bundle(fixture))
    verified = subject.verify_contribution_target_collection(
        config=config, run_dir=result.run_dir
    )
    return subject._contribution_comparison_payload(verified)


@pytest.mark.parametrize("case", sorted(COMPARISON_MISMATCH_CASES))
def test_comparison_categorizes_every_scientific_mismatch(
    case: str, verified_reference_payload: dict[str, Any]
) -> None:
    predicate, reason_fragment = COMPARISON_MISMATCH_CASES[case]
    first = copy.deepcopy(verified_reference_payload)
    second = copy.deepcopy(verified_reference_payload)
    baseline = subject._compare_contribution_target_payloads(first=first, second=second)
    assert baseline.scientifically_equivalent is True
    assert baseline.reasons == ()

    _mutate_comparison_payload(second, case)
    comparison = subject._compare_contribution_target_payloads(first=first, second=second)
    assert comparison.scientifically_equivalent is False
    assert getattr(comparison, predicate) is False
    assert any(reason_fragment in reason for reason in comparison.reasons)
    untouched = [
        field.name
        for field in dataclasses.fields(comparison)
        if field.name
        not in {"scientifically_equivalent", "reasons", "file_byte_equal", predicate}
    ]
    assert all(getattr(comparison, name) is True for name in untouched)


def test_comparison_reasons_name_the_sample_depth_and_family(
    verified_reference_payload: dict[str, Any]
) -> None:
    first = copy.deepcopy(verified_reference_payload)
    second = copy.deepcopy(verified_reference_payload)
    stable_id = sorted(second["records_by_id"])[0]
    depth_key = sorted(second["records_by_id"][stable_id]["depth_targets"])[0]
    _mutate_comparison_payload(second, "raw_utility_drift")
    comparison = subject._compare_contribution_target_payloads(first=first, second=second)
    reason = next(reason for reason in comparison.reasons if "raw utility" in reason)
    assert stable_id in reason
    assert depth_key in reason
    assert "gt_localization" in reason


def test_verifier_rejects_a_nonzero_manifest_teacher_forward_count(
    tmp_path: Path, input_bundle: Any
) -> None:
    config = _official_config(tmp_path)
    result = _materialize(tmp_path, config, input_bundle)
    manifest_path = result.run_dir / subject.FINAL_MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["teacher_forward_count"] == 0
    payload["teacher_forward_count"] = 1
    tampered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    manifest_path.write_text(tampered, encoding="utf-8")
    (result.run_dir / subject.FINAL_MANIFEST_RECEIPT_NAME).write_text(
        hashlib.sha256(tampered.encode("utf-8")).hexdigest() + "\n", encoding="utf-8"
    )
    assert subject.verify_final_manifest_receipt(result.run_dir)
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.verify_contribution_target_collection(
            config=config, run_dir=result.run_dir
        )
    assert _error_code(excinfo) == "B2_CONTRIBUTION_TEACHER_FORWARD_NONZERO"


def _gate_enabled_official_config(tmp_path: Path) -> Any:
    return dataclasses.replace(
        _official_config(tmp_path),
        repository_identity_gate_enabled=True,
        expected_contribution_contract_tag="b2-contribution-target-contract-v1",
        expected_contribution_contract_commit="29591668c3228f6cebd7fd923ae1c39c6dad49bc",
    )


@pytest.mark.parametrize("head", ["parent", "sibling", "unrelated"])
def test_official_api_rejects_a_non_descendant_head(
    tmp_path: Path, input_bundle: Any, head: str
) -> None:
    repository = _clone_contract_repository(tmp_path, name=f"repository-{head}")
    contract_commit = "29591668c3228f6cebd7fd923ae1c39c6dad49bc"
    parent_commit = _git(repository, "rev-parse", f"{contract_commit}^1")
    if head == "parent":
        _git(repository, "checkout", "--quiet", "--detach", parent_commit)
    elif head == "sibling":
        _git(repository, "checkout", "--quiet", "--detach", parent_commit)
        (repository / "sibling-branch.txt").write_text("sibling\n", encoding="utf-8")
        _git(repository, "add", "sibling-branch.txt")
        _git(repository, "commit", "--quiet", "-m", "sibling commit")
    else:
        _git(repository, "checkout", "--quiet", "--orphan", "unrelated-history")
        _git(repository, "rm", "-r", "-f", "-q", ".")
        (repository / "unrelated.txt").write_text("unrelated\n", encoding="utf-8")
        _git(repository, "add", "unrelated.txt")
        _git(repository, "commit", "--quiet", "-m", "unrelated root commit")
    assert _git(repository, "status", "--porcelain", "--untracked-files=all") == ""
    assert _git(repository, "rev-parse", f"{contract_commit}^{{commit}}") == contract_commit

    config = _gate_enabled_official_config(tmp_path)
    run_dir = tmp_path / f"run-{head}"
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.materialize_contribution_target_collection(
            config=config,
            inputs=input_bundle,
            output_run_dir=run_dir,
            expected_plan_sha256="0" * 64,
            repository_root=repository,
        )
    assert _error_code(excinfo) == "B2_CONTRIBUTION_HEAD_NOT_DESCENDANT"
    assert not run_dir.exists()
    with pytest.raises(subject.ContributionTargetError) as excinfo:
        subject.verify_contribution_repository_identity(
            config=config, repository_root=repository
        )
    assert _error_code(excinfo) == "B2_CONTRIBUTION_HEAD_NOT_DESCENDANT"

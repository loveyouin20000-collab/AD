"""B2-04A Story 1 TDD RED: mathematical contracts for dual contribution targets.

``rad.phase_b.b2_contribution_targets`` does not exist yet, so importing it below
fails the whole module at collection time. That is the expected RED signal.

--------------------------------------------------------------------------------
Contract assumed by every test in this file (Story 1 = mathematics only; no
persistence, no CLI, no plan hash, no teacher checkpoint, no dataset access):

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
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import rad.phase_b.b2_contribution_targets as subject  # RED: module does not exist yet
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


def test_module_avoids_teacher_loading_writes_and_target_domain() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("load_teacher_bundle", "torch.save", "atomic_write", "visa", "mkdir"):
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
    calibrated = _tensor([[0.5, 0.25], [1.0, 0.0]])
    mask = _tensor([[1.0, 0.0], [1.0, 0.0]])
    intersection = 0.5 * 1.0 + 1.0 * 1.0
    expected = (2.0 * intersection) / (1.75 + 2.0 + subject.SOFT_DICE_EPS)
    value = subject.soft_dice(calibrated, mask)
    assert isinstance(value, float)
    assert value == pytest.approx(expected, abs=1e-15)


def test_soft_dice_perfect_overlap_is_almost_one() -> None:
    mask = _tensor([[1.0, 0.0], [1.0, 0.0]])
    value = subject.soft_dice(mask, mask)
    assert value == pytest.approx(1.0, abs=1e-6)
    assert value < 1.0


def test_soft_dice_disjoint_prediction_is_zero() -> None:
    calibrated = _tensor([[0.0, 1.0], [0.0, 1.0]])
    mask = _tensor([[1.0, 0.0], [1.0, 0.0]])
    assert subject.soft_dice(calibrated, mask) == 0.0


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
    assert components.utility == pytest.approx(0.4 + 0.4 * components.soft_dice, abs=1e-15)
    assert components.utility < 0.8


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

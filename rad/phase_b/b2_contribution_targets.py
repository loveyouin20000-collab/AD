"""B2-04A dual contribution-target mathematics, artifact contracts, and dry-run.

Story 1 scope: frozen mathematical contracts. Coalition encoding and equal
average fusion, source-training-only GT map calibration, the frozen GT and
teacher utilities, exact Shapley values, and the positive allocation fallback.

Story 2 scope: the per-sample scientific record schema, the GT map calibration
and Shapley normalization artifacts, the layered scientific identities (split
coverage, collection, plan), and the pure leakage-access helpers a future DLCM
loader will use. Every scientific digest comes from an explicit whitelist over
canonical JSON, so paths, timestamps, Git state, runtime attestation, and
file-byte hashes can never enter a scientific identity.

Story 3 scope: configuration loading, the shared in-memory collection /
dry-run path, atomic dual-hash persistence, the final-manifest receipt, and the
official-materialization gate (disabled for the tracked Gate-C configuration).

The module never loads a teacher checkpoint, never runs a backbone, never
accesses a held-out target-domain dataset, never inspects Git, and never mutates
runtime backend settings. Production math is reused rather than reimplemented:
Pixel AP comes from ``rad.evaluation.paper_metrics._binary_ap``, the full-depth
teacher reference comes from ``rad.models.dlcm.sum_preserving_fusion``, and
tensor provenance digests come from
``rad.phase_b.b2_teacher_cache.canonical_tensor_digest``.

Spatial maps are logically ``[height, width]``. ``as_spatial_map`` also accepts
the teacher-cache shapes ``[1, 1, H, W]`` and ``[1, H, W]`` and always returns a
detached ``float64`` 2-D tensor, so every utility below is evaluated in
``float64`` regardless of the cached storage dtype.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath
from types import MappingProxyType
from typing import Any, NamedTuple, NoReturn

import torch
import torch.nn.functional as functional

import rad.phase_b.b2_teacher_cache as cache_mod
from rad.artifacts import atomic_write_json, refuse_existing_run
from rad.errors import OutputProtectionError
from rad.evaluation import paper_metrics
from rad.models import dlcm

FLOAT64 = torch.float64
TAU = 1e-12
SOFT_DICE_EPS = 1e-6
EFFICIENCY_TOLERANCE = 1e-12
ALLOCATION_TIE_TOLERANCE = 1e-12
ALLOCATION_SUM_TOLERANCE = 1e-12
MASK_BINARIZATION_THRESHOLD = 0.5
TOP_PERCENT_FRACTION = 0.01

# Mirrors of the accepted Gate C configuration. Reusable logic never assumes
# these values: candidate layers and prediction depths are always arguments.
DEFAULT_CANDIDATE_LAYERS: tuple[int, ...] = (6, 12, 18, 24)
DEFAULT_PREDICTION_DEPTHS: tuple[int, ...] = (12, 18, 24)

GT_CALIBRATION_QUANTILES = (0.01, 0.995)
GT_ABNORMAL_WEIGHTS = (0.4, 0.4, 0.2)
GT_NORMAL_WEIGHTS = (0.7, 0.3)
BACKGROUND_PENALTY_WEIGHTS = (0.7, 0.3)
TEACHER_UTILITY_WEIGHTS = (0.5, 0.5)

COALITION_CONTRACT_VERSION = "b2_04a_coalition_v1"
UTILITY_CONTRACT_VERSION = "b2_04a_utility_v1"
SHAPLEY_CONTRACT_VERSION = "b2_04a_shapley_v1"
ALLOCATION_CONTRACT_VERSION = "b2_04a_allocation_v1"
RECORD_CONTRACT_VERSION = "b2_04a_record_v1"
CALIBRATION_CONTRACT_VERSION = "b2_04a_calibration_v1"
NORMALIZATION_CONTRACT_VERSION = "b2_04a_normalization_v1"
COLLECTION_CONTRACT_VERSION = "b2_04a_collection_v1"
PLAN_CONTRACT_VERSION = "b2_04a_plan_v1"

RECORD_SCHEMA_VERSION = 1
TARGET_FAMILIES: tuple[str, ...] = ("gt_localization", "teacher_fidelity")
SPLIT_MEMBERSHIPS: tuple[str, ...] = ("training", "calibration", "evaluation")
REQUIRED_SPLIT_COUNTS: Mapping[str, int] = MappingProxyType(
    {"training": 16, "calibration": 8, "evaluation": 8}
)
ACCESS_MODES: tuple[str, ...] = ("training_only", "calibration_only", "evaluation_only")
STATISTICS_DTYPE = "float64"
STANDARD_DEVIATION_DDOF = 0
QUANTILE_RULE = "nearest_rank_ceiling"
PRODUCTION_ARTIFACT_KIND = "production"
TEST_FIXTURE_ARTIFACT_KIND = "test_fixture"

_TEACHER_FUSION_FUNCTION = "rad.models.dlcm.sum_preserving_fusion"
_MASK_SOURCE_ANOMALOUS = "production_gt_mask"
_MASK_SOURCE_NORMAL = "normal_all_zero_mask"
_MASK_ALIGNMENT_MODE = "nearest"

_TRAINING_MEMBERSHIP = "training"
_STATISTICS_DTYPE = "float64"


class ContributionTargetError(RuntimeError):
    """A contribution-target contract failure carrying a stable error code."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise ContributionTargetError(code, detail)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: Any) -> bool:
    return _is_int(value) or (isinstance(value, float) and math.isfinite(value))


# ---------------------------------------------------------------------------
# Spatial map normalization
# ---------------------------------------------------------------------------


def as_spatial_map(tensor: Any, *, role: str = "map") -> torch.Tensor:
    """Normalize one anomaly map to a detached ``float64`` ``[height, width]`` tensor."""

    if not isinstance(tensor, torch.Tensor):
        _fail("B2_TARGET_MAP_TYPE_INVALID", f"{role} must be a torch.Tensor")
    if not tensor.is_floating_point():
        _fail("B2_TARGET_MAP_DTYPE_INVALID", f"{role} must be a floating-point tensor")
    squeezed = tensor
    while squeezed.ndim > 2:
        if int(squeezed.shape[0]) != 1:
            _fail(
                "B2_TARGET_MAP_SHAPE_INVALID",
                f"{role} must be a single spatial map, got shape {tuple(tensor.shape)}",
            )
        squeezed = squeezed[0]
    if squeezed.ndim != 2 or any(int(size) < 1 for size in squeezed.shape):
        _fail(
            "B2_TARGET_MAP_SHAPE_INVALID",
            f"{role} must be [height, width], got shape {tuple(tensor.shape)}",
        )
    result = squeezed.detach().to(dtype=FLOAT64)
    if not bool(torch.isfinite(result).all()):
        _fail("B2_TARGET_MAP_NONFINITE", f"{role} contains NaN or Inf")
    return result.contiguous()


def _require_same_shape(left: torch.Tensor, right: torch.Tensor, *, code: str, detail: str) -> None:
    if tuple(left.shape) != tuple(right.shape):
        _fail(code, f"{detail}: {tuple(left.shape)} vs {tuple(right.shape)}")


def _require_binary_mask(mask: Any, reference: torch.Tensor) -> torch.Tensor:
    binary = as_spatial_map(mask, role="mask")
    _require_same_shape(
        binary,
        reference,
        code="B2_TARGET_MASK_SHAPE_MISMATCH",
        detail="mask shape does not match the map shape",
    )
    unique = set(binary.unique().tolist())
    if not unique <= {0.0, 1.0}:
        _fail("B2_TARGET_MASK_NOT_BINARY", "mask must be binarized to {0, 1} first")
    return binary


# ---------------------------------------------------------------------------
# Coalition encoding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Coalition:
    """One depth-local coalition identified by its ascending integer bitmask."""

    bitmask: int
    layer_ids: tuple[int, ...]


def validate_player_order(players: Sequence[int]) -> tuple[int, ...]:
    """Return the canonical player tuple; fail closed on order or identity drift."""

    if isinstance(players, str | bytes) or not isinstance(players, Sequence):
        _fail("B2_TARGET_PLAYER_ORDER_INVALID", "players must be an ordered sequence")
    ordered = tuple(players)
    if (
        not ordered
        or any(not _is_int(layer) or layer < 1 for layer in ordered)
        or any(ordered[index] >= ordered[index + 1] for index in range(len(ordered) - 1))
    ):
        _fail(
            "B2_TARGET_PLAYER_ORDER_INVALID",
            f"players must be unique, positive, and ascending, got {ordered}",
        )
    return ordered


def _validate_candidate_layers(candidate_layers: Sequence[int]) -> tuple[int, ...]:
    if isinstance(candidate_layers, str | bytes) or not isinstance(candidate_layers, Sequence):
        _fail("B2_TARGET_CANDIDATE_LAYERS_INVALID", "candidate layers must be a sequence")
    ordered = tuple(candidate_layers)
    if (
        not ordered
        or any(not _is_int(layer) or layer < 1 for layer in ordered)
        or any(ordered[index] >= ordered[index + 1] for index in range(len(ordered) - 1))
    ):
        _fail(
            "B2_TARGET_CANDIDATE_LAYERS_INVALID",
            f"candidate layers must be unique, positive, and ascending, got {ordered}",
        )
    return ordered


def players_for_depth(candidate_layers: Sequence[int], depth: int) -> tuple[int, ...]:
    """Return the configured candidate layers at or below ``depth``, ascending."""

    ordered = _validate_candidate_layers(candidate_layers)
    if not _is_int(depth) or depth < 1:
        _fail("B2_TARGET_DEPTH_INVALID", f"prediction depth must be a positive integer, got {depth!r}")
    players = tuple(layer for layer in ordered if layer <= depth)
    if not players:
        _fail(
            "B2_TARGET_DEPTH_HAS_NO_PLAYERS",
            f"depth {depth} has no candidate-layer players in {ordered}",
        )
    return players


def enumerate_coalitions(players: Sequence[int]) -> tuple[Coalition, ...]:
    """Enumerate all coalitions by ascending bitmask; local bit ``i`` is player ``i``."""

    ordered = validate_player_order(players)
    return tuple(
        Coalition(
            bitmask=bitmask,
            layer_ids=tuple(
                layer for index, layer in enumerate(ordered) if (bitmask >> index) & 1
            ),
        )
        for bitmask in range(1 << len(ordered))
    )


def validate_coalition_order(
    coalitions: Sequence[Coalition],
    players: Sequence[int],
) -> tuple[Coalition, ...]:
    """Fail closed unless the enumeration matches the canonical bitmask order exactly."""

    expected = enumerate_coalitions(players)
    actual = tuple(coalitions)
    if len(actual) != len(expected) or any(
        not isinstance(item, Coalition)
        or item.bitmask != reference.bitmask
        or tuple(item.layer_ids) != reference.layer_ids
        for item, reference in zip(actual, expected, strict=True)
    ):
        _fail(
            "B2_TARGET_COALITION_ORDER_INVALID",
            "coalitions must be the complete enumeration in ascending bitmask order",
        )
    return expected


# ---------------------------------------------------------------------------
# Coalition fusion
# ---------------------------------------------------------------------------


def fuse_equal_average(
    layer_maps: Mapping[int, Any],
    layer_ids: Sequence[int],
    *,
    template: Any | None = None,
) -> torch.Tensor:
    """Fuse coalition members with the equal average, never sum-preserving fusion."""

    if not isinstance(layer_maps, Mapping):
        _fail("B2_TARGET_FUSION_INPUT_INVALID", "layer maps must be a mapping")
    selected = tuple(layer_ids)
    if not selected:
        source = template
        if source is None and layer_maps:
            source = layer_maps[min(layer_maps)]
        if source is None:
            _fail(
                "B2_TARGET_FUSION_TEMPLATE_MISSING",
                "the empty coalition needs an authoritative map template",
            )
        return torch.zeros_like(as_spatial_map(source, role="template"))
    validate_player_order(selected)
    missing = [layer for layer in selected if layer not in layer_maps]
    if missing:
        _fail("B2_TARGET_FUSION_LAYER_MISSING", f"coalition layers {missing} have no map")
    members = [as_spatial_map(layer_maps[layer], role=f"map for layer {layer}") for layer in selected]
    for member in members[1:]:
        _require_same_shape(
            members[0],
            member,
            code="B2_TARGET_FUSION_SHAPE_MISMATCH",
            detail="coalition member shapes differ",
        )
    return torch.stack(members, dim=0).mean(dim=0)


# ---------------------------------------------------------------------------
# Nearest-rank quantiles and source-training-only GT calibration
# ---------------------------------------------------------------------------


def nearest_rank_quantile(sorted_values: Any, q: float) -> float:
    """Nearest-rank ceiling quantile ``Q(q) = x[max(1, ceil(q * N)) - 1]`` in float64."""

    if not _is_real(q) or not 0.0 <= float(q) <= 1.0:
        _fail("B2_TARGET_QUANTILE_LEVEL_INVALID", f"quantile level must be in [0, 1], got {q!r}")
    if (
        not isinstance(sorted_values, torch.Tensor)
        or sorted_values.ndim != 1
        or sorted_values.numel() < 1
        or sorted_values.dtype is not FLOAT64
    ):
        _fail(
            "B2_TARGET_QUANTILE_INPUT_INVALID",
            "quantile input must be a non-empty 1-D float64 tensor",
        )
    values = sorted_values.detach()
    if not bool(torch.isfinite(values).all()):
        _fail("B2_TARGET_QUANTILE_INPUT_INVALID", "quantile input contains NaN or Inf")
    count = int(values.numel())
    if count > 1 and not bool((values[1:] >= values[:-1]).all()):
        _fail("B2_TARGET_QUANTILE_INPUT_INVALID", "quantile input must be sorted ascending")
    rank = max(1, math.ceil(float(q) * count))
    return float(values[min(rank, count) - 1].item())


@dataclass(frozen=True)
class GtCalibrationSample:
    """One source-training sample contributing calibration statistics."""

    stable_sample_id: str
    membership: str
    maps_by_depth: Mapping[int, Mapping[int, Any]]


@dataclass(frozen=True)
class DepthCalibration:
    """Frozen per-depth robust monotonic calibration bounds."""

    depth: int
    players: tuple[int, ...]
    nonempty_coalition_count: int
    training_sample_count: int
    value_count: int
    q_low: float
    q_high: float


@dataclass(frozen=True)
class GtMapCalibration:
    """Depth-independent GT map calibration fitted on source training data only."""

    contract_version: str
    statistics_dtype: str
    q_low_quantile: float
    q_high_quantile: float
    candidate_layers: tuple[int, ...]
    prediction_depths: tuple[int, ...]
    ordered_training_stable_sample_ids: tuple[str, ...]
    by_depth: Mapping[int, DepthCalibration]


def _validate_prediction_depths(prediction_depths: Sequence[int]) -> tuple[int, ...]:
    if isinstance(prediction_depths, str | bytes) or not isinstance(prediction_depths, Sequence):
        _fail("B2_TARGET_PREDICTION_DEPTHS_INVALID", "prediction depths must be a sequence")
    ordered = tuple(prediction_depths)
    if (
        not ordered
        or any(not _is_int(depth) or depth < 1 for depth in ordered)
        or any(ordered[index] >= ordered[index + 1] for index in range(len(ordered) - 1))
    ):
        _fail(
            "B2_TARGET_PREDICTION_DEPTHS_INVALID",
            f"prediction depths must be unique, positive, and ascending, got {ordered}",
        )
    return ordered


def _training_only(samples: Sequence[GtCalibrationSample]) -> tuple[str, ...]:
    if not isinstance(samples, Sequence) or not samples:
        _fail(
            "B2_TARGET_CALIBRATION_SAMPLES_MISSING",
            "GT calibration requires at least one source-training sample",
        )
    seen: list[str] = []
    for sample in samples:
        if not isinstance(sample, GtCalibrationSample):
            _fail(
                "B2_TARGET_CALIBRATION_SAMPLE_INVALID",
                "calibration inputs must be GtCalibrationSample records",
            )
        if not isinstance(sample.stable_sample_id, str) or not sample.stable_sample_id:
            _fail(
                "B2_TARGET_CALIBRATION_SAMPLE_INVALID",
                "calibration sample identity must be a non-empty string",
            )
        if sample.membership != _TRAINING_MEMBERSHIP:
            _fail(
                "B2_TARGET_CALIBRATION_LEAKAGE",
                f"sample {sample.stable_sample_id} has membership {sample.membership!r}; "
                "GT calibration is source-training-only",
            )
        if sample.stable_sample_id in seen:
            _fail(
                "B2_TARGET_CALIBRATION_DUPLICATE_SAMPLE",
                f"sample {sample.stable_sample_id} appears more than once",
            )
        seen.append(sample.stable_sample_id)
    return tuple(seen)


def _depth_maps(sample: GtCalibrationSample, depth: int, players: tuple[int, ...]) -> Mapping[int, Any]:
    maps_by_depth = sample.maps_by_depth
    if not isinstance(maps_by_depth, Mapping) or depth not in maps_by_depth:
        _fail(
            "B2_TARGET_CALIBRATION_MAP_LATTICE_INCOMPLETE",
            f"sample {sample.stable_sample_id} has no maps at depth {depth}",
        )
    layer_maps = maps_by_depth[depth]
    if not isinstance(layer_maps, Mapping) or tuple(sorted(layer_maps)) != players:
        _fail(
            "B2_TARGET_CALIBRATION_MAP_LATTICE_INCOMPLETE",
            f"sample {sample.stable_sample_id} at depth {depth} must provide exactly {players}",
        )
    return layer_maps


def fit_gt_map_calibration(
    samples: Sequence[GtCalibrationSample],
    *,
    candidate_layers: Sequence[int],
    prediction_depths: Sequence[int],
    quantiles: tuple[float, float] = GT_CALIBRATION_QUANTILES,
) -> GtMapCalibration:
    """Fit per-depth nearest-rank calibration bounds over every nonempty coalition."""

    layers = _validate_candidate_layers(candidate_layers)
    depths = _validate_prediction_depths(prediction_depths)
    q_low_level, q_high_level = float(quantiles[0]), float(quantiles[1])
    if not 0.0 <= q_low_level < q_high_level <= 1.0:
        _fail(
            "B2_TARGET_CALIBRATION_QUANTILES_INVALID",
            f"calibration quantile levels must satisfy 0 <= low < high <= 1, got {quantiles}",
        )
    ordered_ids = _training_only(samples)

    by_depth: dict[int, DepthCalibration] = {}
    for depth in depths:
        players = players_for_depth(layers, depth)
        coalitions = [item for item in enumerate_coalitions(players) if item.bitmask]
        chunks: list[torch.Tensor] = []
        spatial_shape: tuple[int, ...] | None = None
        for sample in samples:
            layer_maps = _depth_maps(sample, depth, players)
            for coalition in coalitions:
                fused = fuse_equal_average(layer_maps, coalition.layer_ids)
                if spatial_shape is None:
                    spatial_shape = tuple(fused.shape)
                elif tuple(fused.shape) != spatial_shape:
                    _fail(
                        "B2_TARGET_CALIBRATION_SHAPE_MISMATCH",
                        f"depth {depth} coalition maps must share one shape",
                    )
                chunks.append(fused.reshape(-1))
        pooled = torch.sort(torch.cat(chunks)).values
        q_low = nearest_rank_quantile(pooled, q_low_level)
        q_high = nearest_rank_quantile(pooled, q_high_level)
        if not q_high > q_low:
            _fail(
                "B2_TARGET_CALIBRATION_DEGENERATE",
                f"depth {depth} calibration is degenerate: q_high {q_high} <= q_low {q_low}",
            )
        by_depth[depth] = DepthCalibration(
            depth=depth,
            players=players,
            nonempty_coalition_count=len(coalitions),
            training_sample_count=len(ordered_ids),
            value_count=int(pooled.numel()),
            q_low=q_low,
            q_high=q_high,
        )
    return GtMapCalibration(
        contract_version=UTILITY_CONTRACT_VERSION,
        statistics_dtype=_STATISTICS_DTYPE,
        q_low_quantile=q_low_level,
        q_high_quantile=q_high_level,
        candidate_layers=layers,
        prediction_depths=depths,
        ordered_training_stable_sample_ids=ordered_ids,
        by_depth=MappingProxyType(by_depth),
    )


def apply_gt_calibration(anomaly_map: Any, q_low: float, q_high: float) -> torch.Tensor:
    """Apply the frozen robust monotonic calibration and clip to ``[0, 1]``."""

    if not _is_real(q_low) or not _is_real(q_high) or not float(q_high) > float(q_low):
        _fail(
            "B2_TARGET_CALIBRATION_DEGENERATE",
            f"calibration bounds must be finite with q_high > q_low, got ({q_low!r}, {q_high!r})",
        )
    values = as_spatial_map(anomaly_map, role="anomaly map")
    low = float(q_low)
    span = float(q_high) - low
    return ((values - low) / span).clamp(min=0.0, max=1.0)


def calibration_for_depth(calibration: GtMapCalibration, depth: int) -> DepthCalibration:
    """Return the frozen calibration entry for one prediction depth."""

    if not isinstance(calibration, GtMapCalibration) or depth not in calibration.by_depth:
        _fail(
            "B2_TARGET_CALIBRATION_DEPTH_MISSING",
            f"no fitted calibration for depth {depth!r}",
        )
    return calibration.by_depth[depth]


# ---------------------------------------------------------------------------
# GT mask contract
# ---------------------------------------------------------------------------


def binarize_and_validate_mask(
    mask: Any,
    *,
    is_anomalous: bool,
    map_shape: Sequence[int],
) -> torch.Tensor:
    """Nearest-neighbour align, binarize at 0.5, and validate one production GT mask."""

    if not isinstance(is_anomalous, bool):
        _fail("B2_TARGET_MASK_LABEL_INVALID", "is_anomalous must be an explicit bool")
    shape = tuple(map_shape)
    if len(shape) != 2 or any(not _is_int(size) or size < 1 for size in shape):
        _fail("B2_TARGET_MAP_SHAPE_INVALID", f"map shape must be [height, width], got {shape}")
    values = as_spatial_map(mask, role="mask")
    if tuple(values.shape) != shape:
        values = functional.interpolate(
            values[None, None, :, :],
            size=shape,
            mode="nearest",
        )[0, 0]
    binary = (values > MASK_BINARIZATION_THRESHOLD).to(dtype=FLOAT64)
    positives = int(binary.sum().item())
    if not is_anomalous:
        if positives != 0:
            _fail("B2_TARGET_MASK_NORMAL_NOT_EMPTY", "a normal GT mask must be all zero")
        return binary
    if positives == 0:
        _fail("B2_TARGET_MASK_ANOMALY_MISSING", "an anomalous GT mask must contain anomaly pixels")
    if positives == int(binary.numel()):
        _fail(
            "B2_TARGET_MASK_BACKGROUND_MISSING",
            "an anomalous GT mask must contain background pixels",
        )
    return binary


# ---------------------------------------------------------------------------
# Frozen GT utilities
# ---------------------------------------------------------------------------


def top_k_from_fraction(total: int, fraction: float = TOP_PERCENT_FRACTION) -> int:
    """``K = max(1, ceil(fraction * total))`` over a positive population size."""

    if not _is_int(total) or total < 1:
        _fail("B2_TARGET_TOP_K_POPULATION_INVALID", f"population must be positive, got {total!r}")
    if not _is_real(fraction) or not 0.0 < float(fraction) <= 1.0:
        _fail("B2_TARGET_TOP_K_FRACTION_INVALID", f"fraction must be in (0, 1], got {fraction!r}")
    return max(1, math.ceil(float(fraction) * total))


def _stable_descending_indices(values: torch.Tensor, k: int) -> torch.Tensor:
    """Top-``k`` positions by descending value with row-major (ascending index) ties."""

    order = torch.sort(values, descending=True, stable=True).indices
    return torch.sort(order[:k]).values


@dataclass(frozen=True)
class BackgroundPenaltyComponents:
    """Frozen background-penalty decomposition over the calibrated map."""

    background_pixel_count: int
    k: int
    top1_percent_indices: tuple[int, ...]
    top1_percent_mean: float
    global_mean: float
    penalty: float


@dataclass(frozen=True)
class GtAbnormalUtility:
    """Frozen abnormal GT localization utility and its components."""

    pixel_ap: float
    soft_dice: float
    background_penalty: float
    utility: float


@dataclass(frozen=True)
class GtNormalUtility:
    """Frozen normal GT suppression utility and its components."""

    k: int
    top1_percent_mean: float
    global_mean: float
    utility: float


def soft_dice(calibrated_map: Any, mask: Any) -> float:
    """Soft Dice on the calibrated map.

    Authoritative formula places ``eps = 1e-6`` in both the numerator and the
    linear denominator::

        (2 * sum(pred * mask) + eps) / (sum(pred) + sum(mask) + eps)
    """

    prediction = as_spatial_map(calibrated_map, role="calibrated map")
    binary = _require_binary_mask(mask, prediction)
    intersection = float((prediction * binary).sum().item())
    numerator = 2.0 * intersection + SOFT_DICE_EPS
    denominator = float(prediction.sum().item()) + float(binary.sum().item()) + SOFT_DICE_EPS
    return float(numerator / denominator)


def pixel_ap_raw(raw_map: Any, mask: Any) -> float:
    """Pixel AP on the raw map, delegated to the production ``_binary_ap``."""

    scores = as_spatial_map(raw_map, role="raw map")
    binary = _require_binary_mask(mask, scores)
    y_true = binary.reshape(-1).numpy()
    y_score = scores.reshape(-1).numpy()
    return float(paper_metrics._binary_ap(y_true, y_score))


def background_penalty(calibrated_map: Any, mask: Any) -> BackgroundPenaltyComponents:
    """Top-1% and global background response over the calibrated map."""

    prediction = as_spatial_map(calibrated_map, role="calibrated map")
    binary = _require_binary_mask(mask, prediction)
    flat_prediction = prediction.reshape(-1)
    background_indices = torch.nonzero(binary.reshape(-1) == 0.0, as_tuple=False).reshape(-1)
    count = int(background_indices.numel())
    if count == 0:
        _fail(
            "B2_TARGET_MASK_BACKGROUND_MISSING",
            "the background penalty requires at least one background pixel",
        )
    background_values = flat_prediction[background_indices]
    k = top_k_from_fraction(count)
    selected = background_indices[_stable_descending_indices(background_values, k)]
    top1_percent_mean = float(flat_prediction[selected].mean().item())
    global_mean = float(background_values.mean().item())
    weight_top, weight_global = BACKGROUND_PENALTY_WEIGHTS
    return BackgroundPenaltyComponents(
        background_pixel_count=count,
        k=k,
        top1_percent_indices=tuple(int(index) for index in selected.tolist()),
        top1_percent_mean=top1_percent_mean,
        global_mean=global_mean,
        penalty=float(weight_top * top1_percent_mean + weight_global * global_mean),
    )


def gt_utility_abnormal(
    *,
    raw_map: Any,
    calibrated_map: Any,
    mask: Any,
) -> GtAbnormalUtility:
    """``0.4 PixelAP + 0.4 SoftDice - 0.2 P_BG``, never clipped."""

    pixel_ap = pixel_ap_raw(raw_map, mask)
    dice = soft_dice(calibrated_map, mask)
    penalty = background_penalty(calibrated_map, mask).penalty
    weight_ap, weight_dice, weight_penalty = GT_ABNORMAL_WEIGHTS
    return GtAbnormalUtility(
        pixel_ap=pixel_ap,
        soft_dice=dice,
        background_penalty=penalty,
        utility=float(weight_ap * pixel_ap + weight_dice * dice - weight_penalty * penalty),
    )


def gt_utility_normal(*, calibrated_map: Any) -> GtNormalUtility:
    """``1 - (0.7 Top1%Mean + 0.3 GlobalMean)`` over the whole calibrated map."""

    prediction = as_spatial_map(calibrated_map, role="calibrated map")
    flat = prediction.reshape(-1)
    k = top_k_from_fraction(int(flat.numel()))
    selected = _stable_descending_indices(flat, k)
    top1_percent_mean = float(flat[selected].mean().item())
    global_mean = float(flat.mean().item())
    weight_top, weight_global = GT_NORMAL_WEIGHTS
    return GtNormalUtility(
        k=k,
        top1_percent_mean=top1_percent_mean,
        global_mean=global_mean,
        utility=float(1.0 - (weight_top * top1_percent_mean + weight_global * global_mean)),
    )


# ---------------------------------------------------------------------------
# Full-depth teacher reference
# ---------------------------------------------------------------------------


def reconstruct_full_depth_teacher(
    maps_by_layer: Mapping[int, Any],
    *,
    candidate_layers: Sequence[int],
) -> torch.Tensor:
    """Rebuild the full-depth teacher map through the production sum-preserving fusion."""

    layers = _validate_candidate_layers(candidate_layers)
    if not isinstance(maps_by_layer, Mapping) or tuple(sorted(maps_by_layer)) != layers:
        _fail(
            "B2_TARGET_TEACHER_LAYER_SET_INVALID",
            f"the teacher reference requires exactly the candidate layers {layers}",
        )
    members: list[torch.Tensor] = []
    for layer in layers:
        tensor = maps_by_layer[layer]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 4
            or not tensor.is_floating_point()
        ):
            _fail(
                "B2_TARGET_TEACHER_MAP_SHAPE_INVALID",
                f"teacher map for layer {layer} must be a floating-point [batch, channel, height, width] tensor",
            )
        if members and tuple(tensor.shape) != tuple(members[0].shape):
            _fail(
                "B2_TARGET_TEACHER_MAP_SHAPE_INVALID",
                f"teacher map for layer {layer} has shape {tuple(tensor.shape)}",
            )
        if not bool(torch.isfinite(tensor).all()):
            _fail("B2_TARGET_MAP_NONFINITE", f"teacher map for layer {layer} contains NaN or Inf")
        members.append(tensor)
    stacked = torch.stack(members, dim=1)
    valid_mask = torch.ones(stacked.shape[:2], dtype=torch.bool, device=stacked.device)
    weights = valid_mask.to(stacked.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True)
    fused = dlcm.sum_preserving_fusion(stacked, weights, valid_mask)
    expected_shape = tuple(stacked.shape[0:1] + stacked.shape[2:])
    if tuple(fused.shape) != expected_shape:
        _fail(
            "B2_TARGET_TEACHER_SHAPE_MISMATCH",
            f"fused teacher shape {tuple(fused.shape)} is not {expected_shape}",
        )
    if not bool(torch.isfinite(fused).all()):
        _fail("B2_TARGET_MAP_NONFINITE", "the reconstructed teacher map contains NaN or Inf")
    return fused


def verify_full_depth_teacher_bitexact(reconstructed: Any, cached: Any) -> None:
    """Fail closed unless the reconstruction equals the cached teacher map bit-exactly."""

    if not isinstance(reconstructed, torch.Tensor) or not isinstance(cached, torch.Tensor):
        _fail("B2_TARGET_TEACHER_MAP_SHAPE_INVALID", "teacher references must be tensors")
    if tuple(reconstructed.shape) != tuple(cached.shape):
        _fail(
            "B2_TARGET_TEACHER_SHAPE_MISMATCH",
            f"teacher shape {tuple(reconstructed.shape)} != cached {tuple(cached.shape)}",
        )
    if reconstructed.dtype is not cached.dtype:
        _fail(
            "B2_TARGET_TEACHER_DTYPE_MISMATCH",
            f"teacher dtype {reconstructed.dtype} != cached {cached.dtype}",
        )
    left = reconstructed.detach().reshape(-1)
    right = cached.detach().reshape(-1)
    mismatches = torch.nonzero(left != right, as_tuple=False).reshape(-1)
    if int(mismatches.numel()) == 0:
        return None
    first = int(mismatches[0].item())
    difference = (left.to(FLOAT64) - right.to(FLOAT64)).abs()
    max_abs_diff = float(difference.max().item())
    height, width = int(reconstructed.shape[-2]), int(reconstructed.shape[-1])
    row = (first // width) % height
    column = first % width
    _fail(
        "B2_TARGET_TEACHER_NOT_BITEXACT",
        "reconstructed full-depth teacher differs from the cached map: "
        f"max_abs_diff={max_abs_diff:.17g} first_flat_index={first} row={row} col={column}",
    )


# ---------------------------------------------------------------------------
# Teacher utility
# ---------------------------------------------------------------------------


class SpearmanFidelity(NamedTuple):
    """Raw Spearman correlation and its ``[0, 1]`` fidelity rescaling."""

    raw: float
    fidelity: float


@dataclass(frozen=True)
class TeacherUtilityComponents:
    """Frozen teacher-fidelity utility and its components."""

    spearman_raw: float
    spearman_fidelity: float
    top1_overlap: float
    utility: float


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Average (mid) ranks of a 1-D float64 tensor, ties shared deterministically."""

    _unique, inverse, counts = torch.unique(values, return_inverse=True, return_counts=True)
    cumulative = torch.cumsum(counts, dim=0)
    first = cumulative - counts
    average = (first + 1 + cumulative).to(FLOAT64) / 2.0
    return average[inverse]


def _teacher_pair(anomaly_map: Any, teacher_map: Any) -> tuple[torch.Tensor, torch.Tensor]:
    left = as_spatial_map(anomaly_map, role="anomaly map")
    right = as_spatial_map(teacher_map, role="teacher map")
    _require_same_shape(
        left,
        right,
        code="B2_TARGET_TEACHER_SHAPE_MISMATCH",
        detail="anomaly and teacher map shapes differ",
    )
    return left.reshape(-1), right.reshape(-1)


def spearman_fidelity(anomaly_map: Any, teacher_map: Any) -> SpearmanFidelity:
    """Average-rank Spearman with the frozen degeneracy rules, plus its fidelity."""

    left, right = _teacher_pair(anomaly_map, teacher_map)
    left_constant = bool((left.amin() == left.amax()).item())
    right_constant = bool((right.amin() == right.amax()).item())
    if left_constant and right_constant:
        raw = 1.0
    elif left_constant or right_constant:
        raw = 0.0
    else:
        left_ranks = _average_ranks(left)
        right_ranks = _average_ranks(right)
        left_centered = left_ranks - left_ranks.mean()
        right_centered = right_ranks - right_ranks.mean()
        numerator = float((left_centered * right_centered).sum().item())
        denominator = float(
            torch.sqrt((left_centered * left_centered).sum() * (right_centered * right_centered).sum()).item()
        )
        if not denominator > 0.0:
            _fail("B2_TARGET_SPEARMAN_DEGENERATE", "rank variance vanished for non-constant maps")
        raw = min(1.0, max(-1.0, numerator / denominator))
    return SpearmanFidelity(raw=float(raw), fidelity=float((raw + 1.0) / 2.0))


def top1_overlap(anomaly_map: Any, teacher_map: Any) -> float:
    """Top-1% intersection over ``K`` with stable descending selection."""

    left, right = _teacher_pair(anomaly_map, teacher_map)
    k = top_k_from_fraction(int(left.numel()))
    left_top = set(int(index) for index in _stable_descending_indices(left, k).tolist())
    right_top = set(int(index) for index in _stable_descending_indices(right, k).tolist())
    return float(len(left_top & right_top) / k)


def teacher_utility(anomaly_map: Any, teacher_map: Any) -> TeacherUtilityComponents:
    """``0.5 * SpearmanFidelity + 0.5 * Top1%Overlap`` on raw maps only."""

    correlation = spearman_fidelity(anomaly_map, teacher_map)
    overlap = top1_overlap(anomaly_map, teacher_map)
    weight_fidelity, weight_overlap = TEACHER_UTILITY_WEIGHTS
    return TeacherUtilityComponents(
        spearman_raw=correlation.raw,
        spearman_fidelity=correlation.fidelity,
        top1_overlap=overlap,
        utility=float(weight_fidelity * correlation.fidelity + weight_overlap * overlap),
    )


# ---------------------------------------------------------------------------
# Empty-coalition centering, exact Shapley, allocation
# ---------------------------------------------------------------------------


def _validate_utility_domain(utilities: Mapping[int, Any], *, player_count: int | None = None) -> int:
    if not isinstance(utilities, Mapping) or not utilities:
        _fail("B2_TARGET_COALITION_DOMAIN_INVALID", "the coalition domain must be non-empty")
    size = len(utilities)
    if size & (size - 1) or set(utilities) != set(range(size)):
        _fail(
            "B2_TARGET_COALITION_DOMAIN_INVALID",
            "coalition bitmasks must be exactly 0..2**n - 1",
        )
    if player_count is not None and size != 1 << player_count:
        _fail(
            "B2_TARGET_COALITION_DOMAIN_INVALID",
            f"{size} coalitions cannot describe a {player_count}-player game",
        )
    for bitmask in range(size):
        value = utilities[bitmask]
        if not _is_real(value):
            _fail(
                "B2_TARGET_UTILITY_NONFINITE",
                f"utility for coalition {bitmask} is not a finite real number",
            )
    return size


def center_utilities(raw_by_bitmask: Mapping[int, Any]) -> Mapping[int, float]:
    """Center natural utilities so that ``v(empty) = 0`` exactly."""

    size = _validate_utility_domain(raw_by_bitmask)
    baseline = float(raw_by_bitmask[0])
    centered = {bitmask: float(raw_by_bitmask[bitmask]) - baseline for bitmask in range(size)}
    if centered[0] != 0.0:
        _fail("B2_TARGET_CENTERING_NOT_APPLIED", "centered empty-coalition value must be zero")
    return MappingProxyType(centered)


def exact_shapley(
    players: Sequence[int],
    centered_by_bitmask: Mapping[int, Any],
) -> Mapping[int, float]:
    """Exact enumeration Shapley values in float64, keyed by candidate layer."""

    ordered = validate_player_order(players)
    count = len(ordered)
    _validate_utility_domain(centered_by_bitmask, player_count=count)
    if float(centered_by_bitmask[0]) != 0.0:
        _fail(
            "B2_TARGET_CENTERING_NOT_APPLIED",
            "exact Shapley requires empty-coalition-centered utilities",
        )
    total_permutations = float(math.factorial(count))
    phi: dict[int, float] = {}
    for position, player in enumerate(ordered):
        bit = 1 << position
        contribution = 0.0
        for bitmask in range(1 << count):
            if bitmask & bit:
                continue
            size = bin(bitmask).count("1")
            weight = (
                float(math.factorial(size)) * float(math.factorial(count - size - 1))
            ) / total_permutations
            marginal = float(centered_by_bitmask[bitmask | bit]) - float(centered_by_bitmask[bitmask])
            contribution += weight * marginal
        phi[player] = float(contribution)
    return MappingProxyType(phi)


def _phi_values(phi: Mapping[int, Any] | Sequence[float]) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if isinstance(phi, Mapping):
        if not phi:
            _fail("B2_TARGET_ALLOCATION_PLAYERS_MISSING", "the player set is empty")
        keys = tuple(sorted(phi))
        if any(not _is_int(key) for key in keys):
            _fail("B2_TARGET_ALLOCATION_PLAYERS_INVALID", "player keys must be integers")
        values = tuple(phi[key] for key in keys)
    elif isinstance(phi, Sequence) and not isinstance(phi, str | bytes):
        if not phi:
            _fail("B2_TARGET_ALLOCATION_PLAYERS_MISSING", "the player set is empty")
        keys = tuple(range(len(phi)))
        values = tuple(phi)
    else:
        _fail("B2_TARGET_ALLOCATION_PLAYERS_INVALID", "Shapley values must be a mapping or sequence")
    for key, value in zip(keys, values, strict=True):
        if not _is_real(value):
            _fail("B2_TARGET_SHAPLEY_NONFINITE", f"Shapley value for player {key} is not finite")
    return keys, tuple(float(value) for value in values)


def efficiency_residual(phi: Mapping[int, Any] | Sequence[float], v_full: float) -> float:
    """``|sum(phi) - v(N)|`` computed in deterministic ascending player order."""

    if not _is_real(v_full):
        _fail("B2_TARGET_UTILITY_NONFINITE", "the grand-coalition value must be finite")
    _keys, values = _phi_values(phi)
    total = 0.0
    for value in values:
        total += value
    return float(abs(total - float(v_full)))


def require_shapley_efficiency(
    phi: Mapping[int, Any] | Sequence[float],
    v_full: float,
    *,
    tolerance: float = EFFICIENCY_TOLERANCE,
) -> float:
    """Return the efficiency residual; fail closed above ``tolerance``."""

    residual = efficiency_residual(phi, v_full)
    if residual > float(tolerance):
        _fail(
            "B2_TARGET_SHAPLEY_EFFICIENCY_VIOLATION",
            f"efficiency residual {residual:.17g} exceeds tolerance {float(tolerance):.17g}",
        )
    return residual


def positive_allocation(
    phi: Mapping[int, Any] | Sequence[float],
    tau: float = TAU,
) -> Mapping[int, float]:
    """Renormalize positive players, or fall back to minimum-harm equal ties."""

    if not _is_real(tau) or float(tau) < 0.0:
        _fail("B2_TARGET_ALLOCATION_TAU_INVALID", f"tau must be a finite non-negative float, got {tau!r}")
    keys, values = _phi_values(phi)
    threshold = float(tau)
    positive = [index for index, value in enumerate(values) if value > threshold]
    if positive:
        total = 0.0
        for index in positive:
            total += values[index]
        if not total > 0.0:
            _fail("B2_TARGET_ALLOCATION_INVALID", "positive Shapley mass vanished")
        allocation = {
            keys[index]: (values[index] / total if index in set(positive) else 0.0)
            for index in range(len(keys))
        }
    else:
        maximum = max(values)
        winners = [
            index
            for index, value in enumerate(values)
            if maximum - value <= ALLOCATION_TIE_TOLERANCE
        ]
        share = 1.0 / float(len(winners))
        winner_set = set(winners)
        allocation = {
            keys[index]: (share if index in winner_set else 0.0) for index in range(len(keys))
        }
    total_allocation = 0.0
    for key in keys:
        value = allocation[key]
        if not math.isfinite(value) or value < 0.0:
            _fail(
                "B2_TARGET_ALLOCATION_INVALID",
                f"allocation for player {key} must be finite and non-negative",
            )
        total_allocation += value
    if abs(total_allocation - 1.0) > ALLOCATION_SUM_TOLERANCE:
        _fail(
            "B2_TARGET_ALLOCATION_INVALID",
            f"allocation sums to {total_allocation:.17g} instead of 1",
        )
    return MappingProxyType(allocation)


# ---------------------------------------------------------------------------
# Story 2 — scientific records, identities, and leakage-access helpers
# ---------------------------------------------------------------------------


_SHA256_HEX = frozenset("0123456789abcdef")

_NON_SCIENTIFIC_CALIBRATION_KEYS = frozenset(
    {
        "gt_map_calibration_scientific_sha256",
        "artifact_kind",
        "absolute_output_path",
        "timestamp",
        "calibration_file_sha256",
        "git_branch",
        "worktree_path",
        "runtime_attestation_sha256",
        "relative_path",
    }
)
_CALIBRATION_SCIENTIFIC_KEYS = (
    "calibration_contract_version",
    "statistics_dtype",
    "quantile_rule",
    "q_low_quantile",
    "q_high_quantile",
    "candidate_layers",
    "prediction_depths",
    "training_sample_count",
    "ordered_training_stable_sample_ids",
    "source_teacher_record_scientific_sha256_by_id",
    "by_depth",
    "teacher_cache_scientific_sha256",
    "teacher_cache_sample_coverage_sha256",
    "descriptor_collection_scientific_sha256",
    "split_scientific_sha256",
    "checkpoint_sha256",
    "execution_profile_sha256",
    "gt_map_calibration_training_coverage_sha256",
)

_NON_SCIENTIFIC_RECORD_KEYS = frozenset(
    {
        "contribution_target_record_scientific_sha256",
        "artifact_kind",
        "absolute_output_path",
        "relative_record_path",
        "record_file_sha256",
        "timestamp",
        "git_branch",
        "worktree_path",
        "runtime_attestation_sha256",
    }
)
_RECORD_SCIENTIFIC_KEYS = (
    "schema_version",
    "target_record_contract_version",
    "stable_sample_id",
    "split_membership",
    "category",
    "label",
    "anomaly_type",
    "target_families",
    "candidate_layers",
    "prediction_depths",
    "statistics_dtype",
    "coalition_contract_version",
    "utility_contract_version",
    "shapley_contract_version",
    "allocation_contract_version",
    "gt_map_calibration_scientific_sha256",
    "source_teacher_record_scientific_sha256",
    "descriptor_record_scientific_sha256",
    "teacher_cache_scientific_sha256",
    "teacher_cache_sample_coverage_sha256",
    "descriptor_collection_scientific_sha256",
    "split_scientific_sha256",
    "checkpoint_sha256",
    "execution_profile_sha256",
    "mask_provenance",
    "teacher_reference_provenance",
    "depth_targets",
)

_NON_SCIENTIFIC_NORMALIZATION_KEYS = frozenset(
    {
        "shapley_normalization_scientific_sha256",
        "artifact_kind",
        "absolute_output_path",
        "timestamp",
        "normalization_file_sha256",
        "git_branch",
        "worktree_path",
        "runtime_attestation_sha256",
    }
)
_NORMALIZATION_SCIENTIFIC_KEYS = (
    "normalization_contract_version",
    "statistics_dtype",
    "standard_deviation_ddof",
    "target_families",
    "ordered_training_stable_sample_ids",
    "contribution_target_record_scientific_sha256_by_id",
    "axes",
)

_NON_SCIENTIFIC_PLAN_KEYS = frozenset(
    {
        "contribution_plan_scientific_sha256",
        "absolute_output_path",
        "timestamp",
        "git_branch",
        "worktree_path",
        "runtime_attestation_sha256",
        "plan_file_sha256",
    }
)
_PLAN_SCIENTIFIC_KEYS = (
    "gt_map_calibration_scientific_sha256",
    "contribution_target_sample_coverage_sha256",
    "contribution_target_collection_scientific_sha256",
    "shapley_normalization_scientific_sha256",
    "training_target_coverage_sha256",
    "calibration_target_coverage_sha256",
    "evaluation_target_coverage_sha256",
    "planned_record_count",
    "planned_split_counts",
    "planned_ordered_stable_sample_ids",
    "candidate_layers",
    "prediction_depths",
    "contract_versions",
    "teacher_forward_count",
    "official_materialization_enabled",
    "contribution_target_record_scientific_sha256_by_id",
    "teacher_cache_scientific_sha256",
    "descriptor_collection_scientific_sha256",
    "split_scientific_sha256",
    "checkpoint_sha256",
    "execution_profile_sha256",
)

_ACCESS_MODE_TO_MEMBERSHIP = {
    "training_only": "training",
    "calibration_only": "calibration",
    "evaluation_only": "evaluation",
}


@dataclass(frozen=True)
class MaskProvenance:
    """Caller-supplied mask identity and source metadata."""

    mask_identity: str | None
    mask_source: str
    alignment_mode: str = _MASK_ALIGNMENT_MODE
    binarization_threshold: float = MASK_BINARIZATION_THRESHOLD


@dataclass(frozen=True)
class TeacherReferenceProvenance:
    """Caller-supplied verification of the full-depth teacher reference."""

    cached_full_depth_map_digest: str
    reconstruction_verified: bool
    source_candidate_layers: Sequence[int]


@dataclass(frozen=True)
class UpstreamTargetIdentities:
    """Bound teacher-cache and descriptor scientific identities for one sample."""

    source_teacher_record_scientific_sha256: str
    descriptor_record_scientific_sha256: str
    teacher_cache_scientific_sha256: str
    teacher_cache_sample_coverage_sha256: str
    descriptor_collection_scientific_sha256: str
    split_scientific_sha256: str
    checkpoint_sha256: str
    execution_profile_sha256: str


@dataclass(frozen=True)
class ContributionTargetSample:
    """One sample's inputs for building a contribution-target scientific record."""

    stable_sample_id: str
    split_membership: str
    category: str
    label: int
    anomaly_type: str
    maps_by_depth: Mapping[int, Mapping[int, Any]]
    mask: Any
    teacher_reference_map: Any


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_HEX for character in value)
    )


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _project_scientific(
    payload: Mapping[str, Any],
    *,
    whitelist: Sequence[str],
    ignored: frozenset[str],
    schema_code: str,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        _fail(schema_code, "scientific payload must be a mapping")
    unknown = [key for key in payload if key not in whitelist and key not in ignored]
    if unknown:
        _fail(schema_code, f"undeclared scientific fields: {sorted(unknown)}")
    missing = [key for key in whitelist if key not in payload]
    if missing:
        _fail(schema_code, f"missing scientific fields: {missing}")
    return {key: payload[key] for key in whitelist}


def require_production_artifact_kind(payload: Mapping[str, Any]) -> None:
    """Fail closed unless ``artifact_kind`` is the production value."""

    if not isinstance(payload, Mapping):
        _fail("B2_TARGET_ARTIFACT_KIND_INVALID", "artifact payload must be a mapping")
    kind = payload.get("artifact_kind")
    if kind == PRODUCTION_ARTIFACT_KIND:
        return None
    if kind == TEST_FIXTURE_ARTIFACT_KIND:
        _fail(
            "B2_TARGET_TEST_FIXTURE_NOT_ACCEPTED",
            "test_fixture artifacts are never accepted by production official mode",
        )
    _fail("B2_TARGET_ARTIFACT_KIND_INVALID", f"artifact_kind {kind!r} is not accepted")


def full_depth_map_digest(tensor: Any) -> str:
    """Canonical digest of one cached full-depth teacher reference tensor."""

    if not isinstance(tensor, torch.Tensor):
        _fail("B2_TARGET_TEACHER_REFERENCE_DIGEST_MISMATCH", "full-depth map must be a tensor")
    semantics = {
        2: ("height", "width"),
        3: ("batch", "height", "width"),
        4: ("batch", "channel", "height", "width"),
    }.get(int(tensor.ndim))
    if semantics is None:
        _fail(
            "B2_TARGET_TEACHER_REFERENCE_DIGEST_MISMATCH",
            f"unsupported full-depth map rank {int(tensor.ndim)}",
        )
    return cache_mod.canonical_tensor_digest("full_depth_map", tensor, semantics)


def _require_lattice(
    layers: Sequence[int],
    depths: Sequence[int],
    *,
    observed_layers: Any,
    observed_depths: Any,
    code: str,
) -> None:
    expected_layers = list(_validate_candidate_layers(layers))
    expected_depths = list(_validate_prediction_depths(depths))
    if list(observed_layers) != expected_layers or list(observed_depths) != expected_depths:
        _fail(code, "candidate layers or prediction depths drifted from the requested lattice")


def bind_upstream_identities(
    *,
    teacher_record: Mapping[str, Any],
    teacher_record_scientific_sha256: str,
    teacher_cache_scientific_sha256: str,
    teacher_cache_sample_coverage_sha256: str,
    descriptor_record: Mapping[str, Any],
    descriptor_collection_scientific_sha256: str,
    candidate_layers: Sequence[int],
    prediction_depths: Sequence[int],
) -> UpstreamTargetIdentities:
    """Bind and cross-check teacher-record and descriptor-record identities."""

    if not isinstance(teacher_record, Mapping) or not isinstance(descriptor_record, Mapping):
        _fail("B2_TARGET_UPSTREAM_HASH_INVALID", "teacher and descriptor records must be mappings")
    for digest, label in (
        (teacher_record_scientific_sha256, "teacher record"),
        (teacher_cache_scientific_sha256, "teacher cache"),
        (teacher_cache_sample_coverage_sha256, "teacher coverage"),
        (descriptor_collection_scientific_sha256, "descriptor collection"),
    ):
        if not _is_sha256(digest):
            _fail("B2_TARGET_UPSTREAM_HASH_INVALID", f"{label} hash is invalid")
    descriptor_record_hash = descriptor_record.get("descriptor_record_scientific_sha256")
    if not _is_sha256(descriptor_record_hash):
        _fail("B2_TARGET_UPSTREAM_HASH_INVALID", "descriptor record hash is invalid")

    if descriptor_record.get("source_teacher_record_scientific_sha256") != (
        teacher_record_scientific_sha256
    ):
        _fail(
            "B2_TARGET_UPSTREAM_TEACHER_MISMATCH",
            "descriptor record is not anchored to the supplied teacher record",
        )
    if descriptor_record.get("teacher_cache_scientific_sha256") != teacher_cache_scientific_sha256:
        _fail(
            "B2_TARGET_UPSTREAM_TEACHER_MISMATCH",
            "descriptor record teacher-cache identity drifted",
        )

    teacher_id = teacher_record.get("stable_sample_id")
    descriptor_id = descriptor_record.get("stable_sample_id")
    if (
        not isinstance(teacher_id, str)
        or not teacher_id
        or teacher_id != descriptor_id
    ):
        _fail(
            "B2_TARGET_UPSTREAM_SAMPLE_MISMATCH",
            "teacher and descriptor stable sample IDs disagree",
        )

    teacher_membership = teacher_record.get("membership")
    descriptor_membership = descriptor_record.get("split_membership")
    if teacher_membership != descriptor_membership:
        _fail(
            "B2_TARGET_UPSTREAM_SPLIT_MISMATCH",
            "teacher membership and descriptor split_membership disagree",
        )

    for field, code in (
        ("split_scientific_sha256", "B2_TARGET_UPSTREAM_IDENTITY_MISMATCH"),
        ("checkpoint_sha256", "B2_TARGET_UPSTREAM_IDENTITY_MISMATCH"),
        ("execution_profile_sha256", "B2_TARGET_UPSTREAM_IDENTITY_MISMATCH"),
    ):
        teacher_value = teacher_record.get(field)
        descriptor_value = descriptor_record.get(field)
        if not _is_sha256(teacher_value) or not _is_sha256(descriptor_value):
            _fail("B2_TARGET_UPSTREAM_HASH_INVALID", f"{field} is invalid")
        if teacher_value != descriptor_value:
            _fail(code, f"{field} drifted between teacher and descriptor records")

    _require_lattice(
        candidate_layers,
        prediction_depths,
        observed_layers=teacher_record.get("candidate_layers", ()),
        observed_depths=teacher_record.get("prediction_depths", ()),
        code="B2_TARGET_UPSTREAM_LATTICE_MISMATCH",
    )
    _require_lattice(
        candidate_layers,
        prediction_depths,
        observed_layers=descriptor_record.get("candidate_layers", ()),
        observed_depths=descriptor_record.get("prediction_depths", ()),
        code="B2_TARGET_UPSTREAM_LATTICE_MISMATCH",
    )

    return UpstreamTargetIdentities(
        source_teacher_record_scientific_sha256=str(teacher_record_scientific_sha256),
        descriptor_record_scientific_sha256=str(descriptor_record_hash),
        teacher_cache_scientific_sha256=str(teacher_cache_scientific_sha256),
        teacher_cache_sample_coverage_sha256=str(teacher_cache_sample_coverage_sha256),
        descriptor_collection_scientific_sha256=str(descriptor_collection_scientific_sha256),
        split_scientific_sha256=str(teacher_record["split_scientific_sha256"]),
        checkpoint_sha256=str(teacher_record["checkpoint_sha256"]),
        execution_profile_sha256=str(teacher_record["execution_profile_sha256"]),
    )


def gt_map_calibration_scientific_sha256(artifact: Mapping[str, Any]) -> str:
    """Scientific digest of a GT map calibration artifact over the whitelist."""

    projected = _project_scientific(
        artifact,
        whitelist=_CALIBRATION_SCIENTIFIC_KEYS,
        ignored=_NON_SCIENTIFIC_CALIBRATION_KEYS,
        schema_code="B2_TARGET_CALIBRATION_HASH_SCHEMA_INVALID",
    )
    return _canonical_sha256(projected)


def validate_gt_map_calibration_artifact(artifact: Mapping[str, Any]) -> None:
    """Recompute and require the embedded GT calibration scientific hash."""

    claimed = artifact.get("gt_map_calibration_scientific_sha256")
    recomputed = gt_map_calibration_scientific_sha256(artifact)
    if claimed != recomputed:
        _fail(
            "B2_TARGET_CALIBRATION_HASH_MISMATCH",
            "gt_map_calibration_scientific_sha256 does not match scientific content",
        )
    return None


def build_gt_map_calibration_artifact(
    calibration: GtMapCalibration,
    *,
    source_teacher_record_scientific_sha256_by_id: Mapping[str, str],
    teacher_cache_scientific_sha256: str,
    teacher_cache_sample_coverage_sha256: str,
    descriptor_collection_scientific_sha256: str,
    split_scientific_sha256: str,
    checkpoint_sha256: str,
    execution_profile_sha256: str,
    expected_training_count: int,
    artifact_kind: str,
) -> dict[str, Any]:
    """Build the scientific GT map calibration artifact with its digest."""

    if not isinstance(calibration, GtMapCalibration):
        _fail("B2_TARGET_CALIBRATION_COUNT_MISMATCH", "calibration object is invalid")
    ordered_ids = tuple(sorted(calibration.ordered_training_stable_sample_ids))
    if len(ordered_ids) != int(expected_training_count):
        _fail(
            "B2_TARGET_CALIBRATION_COUNT_MISMATCH",
            f"expected {expected_training_count} training samples, got {len(ordered_ids)}",
        )
    if not isinstance(source_teacher_record_scientific_sha256_by_id, Mapping):
        _fail("B2_TARGET_CALIBRATION_COVERAGE_MISMATCH", "teacher hash coverage must be a mapping")
    coverage_ids = set(source_teacher_record_scientific_sha256_by_id)
    if coverage_ids != set(ordered_ids):
        _fail(
            "B2_TARGET_CALIBRATION_COVERAGE_MISMATCH",
            "source teacher hashes must cover exactly the training sample IDs",
        )
    for digest in (
        teacher_cache_scientific_sha256,
        teacher_cache_sample_coverage_sha256,
        descriptor_collection_scientific_sha256,
        split_scientific_sha256,
        checkpoint_sha256,
        execution_profile_sha256,
    ):
        if not _is_sha256(digest):
            _fail("B2_TARGET_UPSTREAM_HASH_INVALID", "calibration upstream hash is invalid")
    teacher_by_id = {
        sample_id: str(source_teacher_record_scientific_sha256_by_id[sample_id])
        for sample_id in ordered_ids
    }
    for digest in teacher_by_id.values():
        if not _is_sha256(digest):
            _fail("B2_TARGET_UPSTREAM_HASH_INVALID", "source teacher record hash is invalid")

    by_depth: dict[str, dict[str, Any]] = {}
    for depth in calibration.prediction_depths:
        entry = calibration.by_depth[depth]
        nonempty = list(range(1, 1 << len(entry.players)))
        by_depth[str(depth)] = {
            "prediction_depth": int(depth),
            "ordered_player_layers": list(entry.players),
            "nonempty_coalition_bitmasks": nonempty,
            "nonempty_coalition_count": int(entry.nonempty_coalition_count),
            "training_sample_count": int(entry.training_sample_count),
            "value_count": int(entry.value_count),
            "q_low": float(entry.q_low),
            "q_high": float(entry.q_high),
        }

    training_coverage = _canonical_sha256(
        {
            "ordered_training_stable_sample_ids": list(ordered_ids),
            "source_teacher_record_scientific_sha256_by_id": teacher_by_id,
        }
    )
    artifact: dict[str, Any] = {
        "calibration_contract_version": CALIBRATION_CONTRACT_VERSION,
        "statistics_dtype": STATISTICS_DTYPE,
        "quantile_rule": QUANTILE_RULE,
        "q_low_quantile": float(calibration.q_low_quantile),
        "q_high_quantile": float(calibration.q_high_quantile),
        "candidate_layers": list(calibration.candidate_layers),
        "prediction_depths": list(calibration.prediction_depths),
        "training_sample_count": len(ordered_ids),
        "ordered_training_stable_sample_ids": list(ordered_ids),
        "source_teacher_record_scientific_sha256_by_id": teacher_by_id,
        "by_depth": by_depth,
        "teacher_cache_scientific_sha256": str(teacher_cache_scientific_sha256),
        "teacher_cache_sample_coverage_sha256": str(teacher_cache_sample_coverage_sha256),
        "descriptor_collection_scientific_sha256": str(descriptor_collection_scientific_sha256),
        "split_scientific_sha256": str(split_scientific_sha256),
        "checkpoint_sha256": str(checkpoint_sha256),
        "execution_profile_sha256": str(execution_profile_sha256),
        "gt_map_calibration_training_coverage_sha256": training_coverage,
        "artifact_kind": str(artifact_kind),
    }
    artifact["gt_map_calibration_scientific_sha256"] = gt_map_calibration_scientific_sha256(artifact)
    return artifact


def contribution_target_record_scientific_sha256(record: Mapping[str, Any]) -> str:
    """Scientific digest of one contribution-target record over the whitelist."""

    projected = _project_scientific(
        record,
        whitelist=_RECORD_SCIENTIFIC_KEYS,
        ignored=_NON_SCIENTIFIC_RECORD_KEYS,
        schema_code="B2_TARGET_RECORD_HASH_SCHEMA_INVALID",
    )
    return _canonical_sha256(projected)


def validate_contribution_target_record(
    record: Mapping[str, Any],
    *,
    candidate_layers: Sequence[int],
    prediction_depths: Sequence[int],
) -> None:
    """Require the embedded record hash and the configured depth lattice."""

    claimed = record.get("contribution_target_record_scientific_sha256")
    recomputed = contribution_target_record_scientific_sha256(record)
    if claimed != recomputed:
        _fail(
            "B2_TARGET_RECORD_HASH_MISMATCH",
            "contribution_target_record_scientific_sha256 does not match scientific content",
        )
    depths = _validate_prediction_depths(prediction_depths)
    depth_targets = record.get("depth_targets")
    if not isinstance(depth_targets, Mapping):
        _fail("B2_TARGET_RECORD_DEPTH_MISSING", "depth_targets must be a mapping")
    for depth in depths:
        if str(depth) not in depth_targets:
            _fail(
                "B2_TARGET_RECORD_DEPTH_MISSING",
                f"record is missing prediction depth {depth}",
            )
    _ = _validate_candidate_layers(candidate_layers)
    return None


def _mask_digest(mask: torch.Tensor) -> str:
    semantics = {
        2: ("height", "width"),
        3: ("batch", "height", "width"),
        4: ("batch", "channel", "height", "width"),
    }[int(mask.ndim)]
    return cache_mod.canonical_tensor_digest("gt_mask", mask.to(dtype=torch.float32), semantics)


def _validate_mask_provenance(
    provenance: MaskProvenance,
    *,
    is_anomalous: bool,
) -> None:
    expected_source = _MASK_SOURCE_ANOMALOUS if is_anomalous else _MASK_SOURCE_NORMAL
    if (
        provenance.mask_source != expected_source
        or provenance.alignment_mode != _MASK_ALIGNMENT_MODE
        or float(provenance.binarization_threshold) != float(MASK_BINARIZATION_THRESHOLD)
    ):
        _fail(
            "B2_TARGET_MASK_PROVENANCE_INVALID",
            "mask provenance does not match the frozen mask contract for this label",
        )


def _gt_components_abnormal(
    *,
    raw_map: torch.Tensor,
    calibrated_map: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, dict[str, float | int]]:
    utility = gt_utility_abnormal(raw_map=raw_map, calibrated_map=calibrated_map, mask=mask)
    penalty = background_penalty(calibrated_map, mask)
    components: dict[str, float | int] = {
        "pixel_ap": float(utility.pixel_ap),
        "soft_dice": float(utility.soft_dice),
        "background_penalty": float(utility.background_penalty),
        "background_pixel_count": int(penalty.background_pixel_count),
        "background_top1_percent_k": int(penalty.k),
        "background_top1_percent_mean": float(penalty.top1_percent_mean),
        "background_global_mean": float(penalty.global_mean),
    }
    return float(utility.utility), components


def _gt_components_normal(*, calibrated_map: torch.Tensor) -> tuple[float, dict[str, float | int]]:
    utility = gt_utility_normal(calibrated_map=calibrated_map)
    components: dict[str, float | int] = {
        "top1_percent_k": int(utility.k),
        "top1_percent_mean": float(utility.top1_percent_mean),
        "global_mean": float(utility.global_mean),
    }
    return float(utility.utility), components


def _teacher_components(
    *,
    fused: torch.Tensor,
    teacher_reference: Any,
) -> tuple[float, dict[str, float]]:
    utility = teacher_utility(fused, teacher_reference)
    components = {
        "spearman_raw": float(utility.spearman_raw),
        "spearman_fidelity": float(utility.spearman_fidelity),
        "top1_overlap": float(utility.top1_overlap),
    }
    return float(utility.utility), components


def _family_block(
    *,
    players: tuple[int, ...],
    raw_by_bitmask: Mapping[int, float],
    utility_mode: str | None,
) -> dict[str, Any]:
    centered = center_utilities(raw_by_bitmask)
    phi = exact_shapley(players, centered)
    residual = require_shapley_efficiency(phi, centered[(1 << len(players)) - 1])
    allocation = positive_allocation(phi)
    block: dict[str, Any] = {
        "empty_coalition_raw_utility": float(raw_by_bitmask[0]),
        "grand_coalition_centered_value": float(centered[(1 << len(players)) - 1]),
        "efficiency_residual": float(residual),
        "raw_signed_shapley_by_layer": {
            str(layer): float(phi[layer]) for layer in players
        },
        "positive_allocation_target_by_layer": {
            str(layer): float(allocation[layer]) for layer in players
        },
    }
    if utility_mode is not None:
        block["utility_mode"] = utility_mode
    return block


def build_contribution_target_record(
    *,
    sample: ContributionTargetSample,
    calibration_artifact: Mapping[str, Any],
    upstream: UpstreamTargetIdentities,
    mask_provenance: MaskProvenance,
    teacher_reference_provenance: TeacherReferenceProvenance,
    candidate_layers: Sequence[int],
    prediction_depths: Sequence[int],
    artifact_kind: str,
) -> dict[str, Any]:
    """Build one dual-family contribution-target scientific record in memory."""

    if not isinstance(sample, ContributionTargetSample):
        _fail("B2_TARGET_RECORD_MEMBERSHIP_INVALID", "sample must be ContributionTargetSample")
    if sample.split_membership not in SPLIT_MEMBERSHIPS:
        _fail(
            "B2_TARGET_RECORD_MEMBERSHIP_INVALID",
            f"split_membership {sample.split_membership!r} is not allowed",
        )
    validate_gt_map_calibration_artifact(calibration_artifact)
    if not teacher_reference_provenance.reconstruction_verified:
        _fail(
            "B2_TARGET_TEACHER_REFERENCE_UNVERIFIED",
            "teacher reference reconstruction must be verified before target construction",
        )
    actual_digest = full_depth_map_digest(sample.teacher_reference_map)
    if actual_digest != teacher_reference_provenance.cached_full_depth_map_digest:
        _fail(
            "B2_TARGET_TEACHER_REFERENCE_DIGEST_MISMATCH",
            "teacher reference digest does not match the supplied map",
        )
    layers = _validate_candidate_layers(candidate_layers)
    depths = _validate_prediction_depths(prediction_depths)
    if tuple(teacher_reference_provenance.source_candidate_layers) != layers:
        _fail(
            "B2_TARGET_TEACHER_REFERENCE_DIGEST_MISMATCH",
            "teacher reference source layers drifted from candidate_layers",
        )

    is_anomalous = int(sample.label) == 1
    _validate_mask_provenance(mask_provenance, is_anomalous=is_anomalous)

    # Establish authoritative spatial shape from the deepest template map.
    deepest = max(depths)
    if not isinstance(sample.maps_by_depth, Mapping) or deepest not in sample.maps_by_depth:
        _fail("B2_TARGET_RECORD_DEPTH_MISSING", f"maps missing for depth {deepest}")
    template_players = players_for_depth(layers, deepest)
    template_maps = sample.maps_by_depth[deepest]
    if not isinstance(template_maps, Mapping) or tuple(sorted(template_maps)) != template_players:
        _fail(
            "B2_TARGET_RECORD_LAYER_SET_INVALID",
            f"depth {deepest} must provide exactly {template_players}",
        )
    template = as_spatial_map(template_maps[template_players[0]], role="template map")
    map_shape = (int(template.shape[0]), int(template.shape[1]))
    binary_mask = binarize_and_validate_mask(
        sample.mask,
        is_anomalous=is_anomalous,
        map_shape=map_shape,
    )
    positive = int((binary_mask > 0.5).sum().item())
    background = int(binary_mask.numel()) - positive
    mask_payload = {
        "mask_identity": mask_provenance.mask_identity,
        "mask_source": mask_provenance.mask_source,
        "alignment_mode": mask_provenance.alignment_mode,
        "binarization_threshold": float(mask_provenance.binarization_threshold),
        "mask_shape": [map_shape[0], map_shape[1]],
        "mask_digest": _mask_digest(
            sample.mask if isinstance(sample.mask, torch.Tensor) else binary_mask
        ),
        "positive_pixel_count": positive,
        "background_pixel_count": background,
    }
    teacher_dtype = (
        str(sample.teacher_reference_map.dtype).replace("torch.", "")
        if isinstance(sample.teacher_reference_map, torch.Tensor)
        else "float32"
    )
    teacher_payload = {
        "fusion_function": _TEACHER_FUSION_FUNCTION,
        "reconstruction_verified": True,
        "source_candidate_layers": list(layers),
        "cached_full_depth_map_digest": str(
            teacher_reference_provenance.cached_full_depth_map_digest
        ),
        "full_depth_map_dtype": teacher_dtype,
    }

    depth_targets: dict[str, Any] = {}
    for depth in depths:
        if depth not in sample.maps_by_depth:
            _fail("B2_TARGET_RECORD_DEPTH_MISSING", f"maps missing for depth {depth}")
        players = players_for_depth(layers, depth)
        layer_maps = sample.maps_by_depth[depth]
        if not isinstance(layer_maps, Mapping) or tuple(sorted(layer_maps)) != players:
            _fail(
                "B2_TARGET_RECORD_LAYER_SET_INVALID",
                f"depth {depth} must provide exactly {players}",
            )
        for layer in players:
            as_spatial_map(layer_maps[layer], role=f"layer {layer} at depth {depth}")

        bounds = calibration_artifact["by_depth"][str(depth)]
        q_low = float(bounds["q_low"])
        q_high = float(bounds["q_high"])
        coalitions = enumerate_coalitions(players)
        raw_gt: dict[int, float] = {}
        raw_teacher: dict[int, float] = {}
        gt_components: dict[int, dict[str, float | int]] = {}
        teacher_components: dict[int, dict[str, float]] = {}
        coalition_table: list[dict[str, Any]] = []
        for coalition in coalitions:
            fused = fuse_equal_average(
                layer_maps, coalition.layer_ids, template=layer_maps[players[0]]
            )
            calibrated = apply_gt_calibration(fused, q_low, q_high)
            if is_anomalous:
                gt_value, gt_comp = _gt_components_abnormal(
                    raw_map=fused, calibrated_map=calibrated, mask=binary_mask
                )
            else:
                gt_value, gt_comp = _gt_components_normal(calibrated_map=calibrated)
            teacher_value, teacher_comp = _teacher_components(
                fused=fused, teacher_reference=sample.teacher_reference_map
            )
            raw_gt[coalition.bitmask] = gt_value
            raw_teacher[coalition.bitmask] = teacher_value
            gt_components[coalition.bitmask] = gt_comp
            teacher_components[coalition.bitmask] = teacher_comp
            centered_placeholder_gt = 0.0
            centered_placeholder_teacher = 0.0
            coalition_table.append(
                {
                    "bitmask": int(coalition.bitmask),
                    "layer_ids": list(coalition.layer_ids),
                    "coalition_size": int(len(coalition.layer_ids)),
                    "gt_localization": {
                        "raw_utility": float(gt_value),
                        "centered_value": centered_placeholder_gt,
                        "utility_components": dict(gt_comp),
                    },
                    "teacher_fidelity": {
                        "raw_utility": float(teacher_value),
                        "centered_value": centered_placeholder_teacher,
                        "utility_components": dict(teacher_comp),
                    },
                }
            )

        centered_gt = center_utilities(raw_gt)
        centered_teacher = center_utilities(raw_teacher)
        for entry in coalition_table:
            bitmask = int(entry["bitmask"])
            entry["gt_localization"]["centered_value"] = float(centered_gt[bitmask])
            entry["teacher_fidelity"]["centered_value"] = float(centered_teacher[bitmask])

        utility_mode = "abnormal" if is_anomalous else "normal"
        depth_targets[str(depth)] = {
            "prediction_depth": int(depth),
            "ordered_player_layers": list(players),
            "coalition_table": coalition_table,
            "gt_localization": _family_block(
                players=players, raw_by_bitmask=raw_gt, utility_mode=utility_mode
            ),
            "teacher_fidelity": _family_block(
                players=players, raw_by_bitmask=raw_teacher, utility_mode=None
            ),
        }

    record: dict[str, Any] = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "target_record_contract_version": RECORD_CONTRACT_VERSION,
        "stable_sample_id": str(sample.stable_sample_id),
        "split_membership": str(sample.split_membership),
        "category": str(sample.category),
        "label": int(sample.label),
        "anomaly_type": str(sample.anomaly_type),
        "target_families": list(TARGET_FAMILIES),
        "candidate_layers": list(layers),
        "prediction_depths": list(depths),
        "statistics_dtype": STATISTICS_DTYPE,
        "coalition_contract_version": COALITION_CONTRACT_VERSION,
        "utility_contract_version": UTILITY_CONTRACT_VERSION,
        "shapley_contract_version": SHAPLEY_CONTRACT_VERSION,
        "allocation_contract_version": ALLOCATION_CONTRACT_VERSION,
        "gt_map_calibration_scientific_sha256": str(
            calibration_artifact["gt_map_calibration_scientific_sha256"]
        ),
        "source_teacher_record_scientific_sha256": (
            upstream.source_teacher_record_scientific_sha256
        ),
        "descriptor_record_scientific_sha256": upstream.descriptor_record_scientific_sha256,
        "teacher_cache_scientific_sha256": upstream.teacher_cache_scientific_sha256,
        "teacher_cache_sample_coverage_sha256": upstream.teacher_cache_sample_coverage_sha256,
        "descriptor_collection_scientific_sha256": (
            upstream.descriptor_collection_scientific_sha256
        ),
        "split_scientific_sha256": upstream.split_scientific_sha256,
        "checkpoint_sha256": upstream.checkpoint_sha256,
        "execution_profile_sha256": upstream.execution_profile_sha256,
        "mask_provenance": mask_payload,
        "teacher_reference_provenance": teacher_payload,
        "depth_targets": depth_targets,
        "artifact_kind": str(artifact_kind),
    }
    record["contribution_target_record_scientific_sha256"] = (
        contribution_target_record_scientific_sha256(record)
    )
    return record


def shapley_normalization_scientific_sha256(artifact: Mapping[str, Any]) -> str:
    """Scientific digest of the Shapley normalization artifact."""

    projected = _project_scientific(
        artifact,
        whitelist=_NORMALIZATION_SCIENTIFIC_KEYS,
        ignored=_NON_SCIENTIFIC_NORMALIZATION_KEYS,
        schema_code="B2_TARGET_NORMALIZATION_HASH_SCHEMA_INVALID",
    )
    return _canonical_sha256(projected)


def compute_shapley_normalization(
    records: Sequence[Mapping[str, Any]],
    *,
    candidate_layers: Sequence[int],
    prediction_depths: Sequence[int],
    expected_training_count: int,
    artifact_kind: str,
) -> dict[str, Any]:
    """Fit training-only Shapley normalization statistics in float64."""

    if not isinstance(records, Sequence) or isinstance(records, str | bytes):
        _fail("B2_TARGET_NORMALIZATION_COUNT_MISMATCH", "records must be a sequence")
    for row in records:
        if not isinstance(row, Mapping):
            _fail("B2_TARGET_NORMALIZATION_MEMBERSHIP_INVALID", "each record must be a mapping")
        if row.get("split_membership") != _TRAINING_MEMBERSHIP:
            _fail(
                "B2_TARGET_NORMALIZATION_MEMBERSHIP_INVALID",
                "Shapley normalization admits training records only",
            )
    if len(records) != int(expected_training_count):
        _fail(
            "B2_TARGET_NORMALIZATION_COUNT_MISMATCH",
            f"expected {expected_training_count} training records, got {len(records)}",
        )
    layers = _validate_candidate_layers(candidate_layers)
    depths = _validate_prediction_depths(prediction_depths)
    ordered = sorted(records, key=lambda row: str(row["stable_sample_id"]))
    ordered_ids = [str(row["stable_sample_id"]) for row in ordered]
    if len(set(ordered_ids)) != len(ordered_ids):
        _fail("B2_TARGET_NORMALIZATION_COUNT_MISMATCH", "duplicate training sample IDs")
    hashes_by_id = {
        str(row["stable_sample_id"]): str(row["contribution_target_record_scientific_sha256"])
        for row in ordered
    }

    axes: dict[str, dict[str, Any]] = {}
    for family in TARGET_FAMILIES:
        axes[family] = {}
        for depth in depths:
            players = players_for_depth(layers, depth)
            layer_entries: list[dict[str, Any]] = []
            for layer in players:
                values = [
                    float(
                        row["depth_targets"][str(depth)][family][
                            "raw_signed_shapley_by_layer"
                        ][str(layer)]
                    )
                    for row in ordered
                ]
                count = len(values)
                mean = math.fsum(values) / float(count)
                variance = math.fsum((value - mean) ** 2 for value in values) / float(count)
                std = math.sqrt(variance)
                layer_entries.append(
                    {
                        "candidate_layer_id": int(layer),
                        "count": int(count),
                        "mean": float(mean),
                        "std": float(std),
                        "minimum": float(min(values)),
                        "maximum": float(max(values)),
                        "zero_variance": bool(std == 0.0),
                    }
                )
            axes[family][str(depth)] = {
                "prediction_depth": int(depth),
                "layers": layer_entries,
            }

    artifact: dict[str, Any] = {
        "normalization_contract_version": NORMALIZATION_CONTRACT_VERSION,
        "statistics_dtype": STATISTICS_DTYPE,
        "standard_deviation_ddof": STANDARD_DEVIATION_DDOF,
        "target_families": list(TARGET_FAMILIES),
        "ordered_training_stable_sample_ids": ordered_ids,
        "contribution_target_record_scientific_sha256_by_id": hashes_by_id,
        "axes": axes,
        "artifact_kind": str(artifact_kind),
    }
    artifact["shapley_normalization_scientific_sha256"] = (
        shapley_normalization_scientific_sha256(artifact)
    )
    return artifact


def standardize_signed_shapley(
    value: Any,
    artifact: Mapping[str, Any],
    *,
    target_family: str,
    prediction_depth: int,
    candidate_layer_id: int,
) -> float:
    """Read-time z-score using frozen normalization statistics."""

    if target_family not in TARGET_FAMILIES:
        _fail("B2_TARGET_FAMILY_INVALID", f"unknown target family {target_family!r}")
    if not _is_real(value):
        _fail("B2_TARGET_NORMALIZATION_AXIS_MISSING", "Shapley value must be a finite real")
    axes = artifact.get("axes")
    if not isinstance(axes, Mapping) or target_family not in axes:
        _fail("B2_TARGET_NORMALIZATION_AXIS_MISSING", "normalization axes are incomplete")
    depth_entry = axes[target_family].get(str(int(prediction_depth)))
    if not isinstance(depth_entry, Mapping):
        _fail("B2_TARGET_NORMALIZATION_AXIS_MISSING", "prediction depth axis is missing")
    layers = depth_entry.get("layers")
    if not isinstance(layers, Sequence):
        _fail("B2_TARGET_NORMALIZATION_AXIS_MISSING", "layer statistics are missing")
    match = None
    for entry in layers:
        if int(entry.get("candidate_layer_id", -1)) == int(candidate_layer_id):
            match = entry
            break
    if match is None:
        _fail(
            "B2_TARGET_NORMALIZATION_AXIS_MISSING",
            f"no statistics for layer {candidate_layer_id} at depth {prediction_depth}",
        )
    mean = float(match["mean"])
    std = float(match["std"])
    if std > 0.0 and not bool(match.get("zero_variance")):
        return float((float(value) - mean) / std)
    return 0.0


def _ordered_records(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(records, key=lambda row: str(row["stable_sample_id"]))


def _split_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {membership: 0 for membership in SPLIT_MEMBERSHIPS}
    for row in records:
        membership = row.get("split_membership")
        if membership not in counts:
            _fail(
                "B2_TARGET_COVERAGE_COUNT_MISMATCH",
                f"unknown split_membership {membership!r}",
            )
        counts[str(membership)] += 1
    return counts


def contribution_target_sample_coverage_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash the ordered 32-sample coverage with required 16/8/8 split counts."""

    if not isinstance(records, Sequence) or isinstance(records, str | bytes):
        _fail("B2_TARGET_COVERAGE_COUNT_MISMATCH", "records must be a sequence")
    counts = _split_counts(records)
    if counts != dict(REQUIRED_SPLIT_COUNTS) or len(records) != sum(REQUIRED_SPLIT_COUNTS.values()):
        _fail(
            "B2_TARGET_COVERAGE_COUNT_MISMATCH",
            f"contribution-target coverage requires {dict(REQUIRED_SPLIT_COUNTS)}, got {counts}",
        )
    ordered = _ordered_records(records)
    payload = {
        "ordered_stable_sample_ids": [str(row["stable_sample_id"]) for row in ordered],
        "split_membership_by_id": {
            str(row["stable_sample_id"]): str(row["split_membership"]) for row in ordered
        },
        "split_counts": dict(REQUIRED_SPLIT_COUNTS),
    }
    return _canonical_sha256(payload)


def _membership_coverage_sha256(
    records: Sequence[Mapping[str, Any]],
    *,
    membership: str,
) -> str:
    if not isinstance(records, Sequence) or isinstance(records, str | bytes):
        _fail("B2_TARGET_COVERAGE_COUNT_MISMATCH", "records must be a sequence")
    for row in records:
        if row.get("split_membership") != membership:
            _fail(
                "B2_TARGET_COVERAGE_MEMBERSHIP_INVALID",
                f"{membership} coverage rejects foreign split membership",
            )
    expected = int(REQUIRED_SPLIT_COUNTS[membership])
    if len(records) != expected:
        _fail(
            "B2_TARGET_COVERAGE_COUNT_MISMATCH",
            f"{membership} coverage requires {expected} records, got {len(records)}",
        )
    ordered = _ordered_records(records)
    return _canonical_sha256(
        {
            "membership": membership,
            "ordered_stable_sample_ids": [str(row["stable_sample_id"]) for row in ordered],
            "contribution_target_record_scientific_sha256_by_id": {
                str(row["stable_sample_id"]): str(
                    row["contribution_target_record_scientific_sha256"]
                )
                for row in ordered
            },
        }
    )


def training_target_coverage_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return _membership_coverage_sha256(records, membership="training")


def calibration_target_coverage_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return _membership_coverage_sha256(records, membership="calibration")


def evaluation_target_coverage_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return _membership_coverage_sha256(records, membership="evaluation")


def contribution_target_collection_scientific_sha256(
    *,
    records: Sequence[Mapping[str, Any]],
    calibration_artifact: Mapping[str, Any],
    normalization: Mapping[str, Any],
    candidate_layers: Sequence[int],
    prediction_depths: Sequence[int],
) -> str:
    """Bind the 32 records to calibration and normalization scientific identities."""

    sample_coverage = contribution_target_sample_coverage_sha256(records)
    ordered = _ordered_records(records)
    training = [row for row in ordered if row["split_membership"] == "training"]
    calibration = [row for row in ordered if row["split_membership"] == "calibration"]
    evaluation = [row for row in ordered if row["split_membership"] == "evaluation"]
    layers = list(_validate_candidate_layers(candidate_layers))
    depths = list(_validate_prediction_depths(prediction_depths))
    first = ordered[0]
    return _canonical_sha256(
        {
            "collection_contract_version": COLLECTION_CONTRACT_VERSION,
            "candidate_layers": layers,
            "prediction_depths": depths,
            "contribution_target_sample_coverage_sha256": sample_coverage,
            "training_target_coverage_sha256": training_target_coverage_sha256(training),
            "calibration_target_coverage_sha256": calibration_target_coverage_sha256(calibration),
            "evaluation_target_coverage_sha256": evaluation_target_coverage_sha256(evaluation),
            "ordered_stable_sample_ids": [str(row["stable_sample_id"]) for row in ordered],
            "contribution_target_record_scientific_sha256_by_id": {
                str(row["stable_sample_id"]): str(
                    row["contribution_target_record_scientific_sha256"]
                )
                for row in ordered
            },
            "gt_map_calibration_scientific_sha256": str(
                calibration_artifact["gt_map_calibration_scientific_sha256"]
            ),
            "shapley_normalization_scientific_sha256": str(
                normalization["shapley_normalization_scientific_sha256"]
            ),
            "teacher_cache_scientific_sha256": str(first["teacher_cache_scientific_sha256"]),
            "descriptor_collection_scientific_sha256": str(
                first["descriptor_collection_scientific_sha256"]
            ),
            "split_scientific_sha256": str(first["split_scientific_sha256"]),
            "checkpoint_sha256": str(first["checkpoint_sha256"]),
            "execution_profile_sha256": str(first["execution_profile_sha256"]),
        }
    )


def _plan_scientific_sha256(plan: Mapping[str, Any]) -> str:
    projected = _project_scientific(
        plan,
        whitelist=_PLAN_SCIENTIFIC_KEYS,
        ignored=_NON_SCIENTIFIC_PLAN_KEYS,
        schema_code="B2_TARGET_RECORD_HASH_SCHEMA_INVALID",
    )
    return _canonical_sha256(projected)


def build_contribution_plan(
    *,
    records: Sequence[Mapping[str, Any]],
    calibration_artifact: Mapping[str, Any],
    normalization: Mapping[str, Any],
    candidate_layers: Sequence[int],
    prediction_depths: Sequence[int],
    official_materialization_enabled: bool = False,
) -> dict[str, Any]:
    """Pure in-memory contribution plan binding all seven layered identities."""

    layers = list(_validate_candidate_layers(candidate_layers))
    depths = list(_validate_prediction_depths(prediction_depths))
    ordered = _ordered_records(records)
    training = [row for row in ordered if row["split_membership"] == "training"]
    calibration = [row for row in ordered if row["split_membership"] == "calibration"]
    evaluation = [row for row in ordered if row["split_membership"] == "evaluation"]
    sample_coverage = contribution_target_sample_coverage_sha256(records)
    training_coverage = training_target_coverage_sha256(training)
    calibration_coverage = calibration_target_coverage_sha256(calibration)
    evaluation_coverage = evaluation_target_coverage_sha256(evaluation)
    collection = contribution_target_collection_scientific_sha256(
        records=records,
        calibration_artifact=calibration_artifact,
        normalization=normalization,
        candidate_layers=layers,
        prediction_depths=depths,
    )
    first = ordered[0]
    hashes_by_id = {
        str(row["stable_sample_id"]): str(row["contribution_target_record_scientific_sha256"])
        for row in ordered
    }
    plan: dict[str, Any] = {
        "gt_map_calibration_scientific_sha256": str(
            calibration_artifact["gt_map_calibration_scientific_sha256"]
        ),
        "contribution_target_sample_coverage_sha256": sample_coverage,
        "contribution_target_collection_scientific_sha256": collection,
        "shapley_normalization_scientific_sha256": str(
            normalization["shapley_normalization_scientific_sha256"]
        ),
        "training_target_coverage_sha256": training_coverage,
        "calibration_target_coverage_sha256": calibration_coverage,
        "evaluation_target_coverage_sha256": evaluation_coverage,
        "planned_record_count": len(ordered),
        "planned_split_counts": {
            "training": len(training),
            "calibration": len(calibration),
            "evaluation": len(evaluation),
        },
        "planned_ordered_stable_sample_ids": [str(row["stable_sample_id"]) for row in ordered],
        "candidate_layers": layers,
        "prediction_depths": depths,
        "contract_versions": {
            "coalition": COALITION_CONTRACT_VERSION,
            "utility": UTILITY_CONTRACT_VERSION,
            "shapley": SHAPLEY_CONTRACT_VERSION,
            "allocation": ALLOCATION_CONTRACT_VERSION,
            "record": RECORD_CONTRACT_VERSION,
            "calibration": CALIBRATION_CONTRACT_VERSION,
            "normalization": NORMALIZATION_CONTRACT_VERSION,
            "collection": COLLECTION_CONTRACT_VERSION,
            "plan": PLAN_CONTRACT_VERSION,
        },
        "teacher_forward_count": 0,
        "official_materialization_enabled": bool(official_materialization_enabled),
        "contribution_target_record_scientific_sha256_by_id": hashes_by_id,
        "teacher_cache_scientific_sha256": str(first["teacher_cache_scientific_sha256"]),
        "descriptor_collection_scientific_sha256": str(
            first["descriptor_collection_scientific_sha256"]
        ),
        "split_scientific_sha256": str(first["split_scientific_sha256"]),
        "checkpoint_sha256": str(first["checkpoint_sha256"]),
        "execution_profile_sha256": str(first["execution_profile_sha256"]),
    }
    plan["contribution_plan_scientific_sha256"] = _plan_scientific_sha256(plan)
    return plan


def load_targets_for_access(
    records: Sequence[Mapping[str, Any]],
    *,
    access_mode: str,
    normalization: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return split-gated read views with raw and standardized Shapley values."""

    if access_mode not in ACCESS_MODES:
        _fail("B2_TARGET_ACCESS_MODE_INVALID", f"unknown access_mode {access_mode!r}")
    claimed = normalization.get("shapley_normalization_scientific_sha256")
    recomputed = shapley_normalization_scientific_sha256(normalization)
    if claimed != recomputed:
        _fail(
            "B2_TARGET_NORMALIZATION_HASH_MISMATCH",
            "normalization artifact scientific hash does not match its content",
        )
    expected_membership = _ACCESS_MODE_TO_MEMBERSHIP[access_mode]
    if not isinstance(records, Sequence) or isinstance(records, str | bytes):
        _fail("B2_TARGET_ACCESS_LEAKAGE", "records must be a sequence")
    for row in records:
        if row.get("split_membership") != expected_membership:
            _fail(
                "B2_TARGET_ACCESS_LEAKAGE",
                f"{access_mode} rejects records outside {expected_membership}",
            )
    ordered = _ordered_records(records)
    views: list[dict[str, Any]] = []
    for row in ordered:
        by_depth: dict[str, Any] = {}
        depth_targets = row["depth_targets"]
        for depth_key, depth_block in depth_targets.items():
            depth = int(depth_key)
            family_views: dict[str, Any] = {}
            for family in TARGET_FAMILIES:
                source = depth_block[family]
                raw = dict(source["raw_signed_shapley_by_layer"])
                allocation = dict(source["positive_allocation_target_by_layer"])
                standardized = {
                    str(layer): standardize_signed_shapley(
                        raw[str(layer)],
                        normalization,
                        target_family=family,
                        prediction_depth=depth,
                        candidate_layer_id=int(layer),
                    )
                    for layer in raw
                }
                family_views[family] = {
                    "raw_signed_shapley_by_layer": raw,
                    "positive_allocation_target_by_layer": allocation,
                    "standardized_signed_shapley_by_layer": standardized,
                }
            by_depth[str(depth)] = family_views
        views.append(
            {
                "stable_sample_id": str(row["stable_sample_id"]),
                "split_membership": str(row["split_membership"]),
                "by_depth": by_depth,
            }
        )
    return views


# ---------------------------------------------------------------------------
# Story 3 — configuration, shared orchestration, dry run, atomic persistence
# ---------------------------------------------------------------------------
#
# Everything below stays inside the same inert boundary as Stories 1 and 2: no
# checkpoint load, no teacher or backbone forward, no dataset adapter, no
# repository inspection, no machine-local path selection, and no backend
# mutation. The only I/O is (a) reading a declared configuration file, (b)
# reading declared upstream artifact roots, and (c) writing a fresh run
# directory during official materialization, which B2-04A keeps disabled.

TRACKED_CONFIGURATION_ID = "b2_contribution_targets_gate_c"
CONTRACT_STAGE = "b2_04a"
OFFICIAL_CONFIGURATION_ID = "b2_contribution_targets_official_v1"
OFFICIAL_CONTRACT_STAGE = "b2_04b"
EXPECTED_CONTRIBUTION_CONTRACT_TAG = "b2-contribution-target-contract-v1"
EXPECTED_CONTRIBUTION_CONTRACT_COMMIT = "29591668c3228f6cebd7fd923ae1c39c6dad49bc"
CONFIG_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1

RECORDS_DIRECTORY = "records"
CALIBRATION_RELATIVE_PATH = "gt_map_calibration.pt"
NORMALIZATION_RELATIVE_PATH = "shapley_normalization.pt"
FINAL_MANIFEST_NAME = "final_manifest.json"
FINAL_MANIFEST_RECEIPT_NAME = "final_manifest.json.sha256"

RECORD_PAYLOAD_KEYS = frozenset(
    {"scientific_record", "contribution_target_record_scientific_sha256"}
)
CALIBRATION_PAYLOAD_KEYS = frozenset(
    {"scientific_calibration_record", "gt_map_calibration_scientific_sha256"}
)
NORMALIZATION_PAYLOAD_KEYS = frozenset(
    {"scientific_normalization_record", "shapley_normalization_scientific_sha256"}
)

SEVEN_LAYERED_IDENTITY_KEYS: tuple[str, ...] = (
    "gt_map_calibration_scientific_sha256",
    "contribution_target_sample_coverage_sha256",
    "contribution_target_collection_scientific_sha256",
    "shapley_normalization_scientific_sha256",
    "training_target_coverage_sha256",
    "calibration_target_coverage_sha256",
    "evaluation_target_coverage_sha256",
)

_RECORD_FILE_HASH_FIELD = "contribution_target_record_file_sha256"

# Pinned Gate-C scientific expectations. Drift in any of them fails closed.
_EXPECTED_EXECUTION_PROFILE_SHA256 = (
    "7af8dba39633743da0380fef9710940cded655f68c9efa8f84f5a52aeddb3c8d"
)
_EXPECTED_SPLIT_SCIENTIFIC_SHA256 = (
    "91570da1fed6d7859d407196b10403581832ae0ff677a1ea7657ca76b91471f0"
)
_EXPECTED_CHECKPOINT_SHA256 = (
    "97bd461163efb96e36cddb1c3adf677e4c4fc2daabb2521021689f30e799b4f4"
)
_EXPECTED_TEACHER_CACHE_SCIENTIFIC_SHA256 = (
    "66d23807e868696a9c4a68ad83399d82df3d33e743a97d97eeb98ac60c0b1b0a"
)
_EXPECTED_TEACHER_CACHE_SAMPLE_COVERAGE_SHA256 = (
    "6e538b902795c377f9992258e307e58b5c0ba0f99cbbe6c3853a81947ca3d76c"
)
_EXPECTED_DESCRIPTOR_COLLECTION_SCIENTIFIC_SHA256 = (
    "eb967822725e730ee2eb8afa3a5c8e28b4657141aa920d6a688ab370c70c6dd9"
)
_EXPECTED_DESCRIPTOR_SAMPLE_COVERAGE_SHA256 = (
    "27d064db21b5c699503be32e414d579bd1aa7158f1d9b141de26555fc79bc6df"
)
_EXPECTED_DESCRIPTOR_NORMALIZATION_SCIENTIFIC_SHA256 = (
    "f77975a94acf87a14b0753aabc9aad6777943ee4e4958b0a2083701cf4528594"
)
_EXPECTED_DESCRIPTOR_NORMALIZATION_TRAINING_COVERAGE_SHA256 = (
    "e940f46bf696d326f8b982f15b8639f81e4548ec31a9b09634729811337e4c90"
)

_EXPECTED_GT_CALIBRATION = {
    "quantile_low": 0.01,
    "quantile_high": 0.995,
    "quantile_algorithm": QUANTILE_RULE,
    "per_depth": True,
    "training_only": True,
}
_EXPECTED_ABNORMAL_GT_WEIGHTS = {
    "pixel_ap": 0.4,
    "soft_dice": 0.4,
    "background_penalty": 0.2,
}
_EXPECTED_FALSE_POSITIVE_WEIGHTS = {"top1": 0.7, "global_mean": 0.3}
_EXPECTED_TEACHER_FIDELITY = {
    "spearman_weight": 0.5,
    "top1_overlap_weight": 0.5,
    "top_fraction": TOP_PERCENT_FRACTION,
    "overlap": "intersection_over_k",
}
_EXPECTED_SHAPLEY = {
    "exact": True,
    "float_dtype": STATISTICS_DTYPE,
    "tolerance": EFFICIENCY_TOLERANCE,
    "empty_coalition_centering": True,
}
_EXPECTED_ALLOCATION = {
    "positive_threshold": TAU,
    "fallback": "minimum_harm_equal_ties",
}
_EXPECTED_COALITION_FUSION = "equal_average"
_EXPECTED_COALITION_ENCODING = "depth_local_bitmask_ascending"
_ACCEPTED_INPUT_ARTIFACT_KINDS = (PRODUCTION_ARTIFACT_KIND, TEST_FIXTURE_ARTIFACT_KIND)


@dataclass(frozen=True)
class ContributionTargetsConfig:
    """The frozen, fully declared contribution-target configuration."""

    schema_version: int
    configuration_id: str
    contract_stage: str
    official_materialization_enabled: bool
    expected_input_artifact_kind: str
    expected_execution_profile_sha256: str
    expected_split_scientific_sha256: str
    expected_checkpoint_sha256: str
    expected_teacher_cache_scientific_sha256: str
    expected_teacher_cache_sample_coverage_sha256: str
    expected_descriptor_collection_scientific_sha256: str
    expected_descriptor_sample_coverage_sha256: str
    expected_descriptor_normalization_scientific_sha256: str
    expected_descriptor_normalization_training_coverage_sha256: str
    candidate_layers: tuple[int, ...]
    prediction_depths: tuple[int, ...]
    target_families: tuple[str, ...]
    coalition_fusion: str
    coalition_encoding: str
    gt_calibration: Mapping[str, Any]
    abnormal_gt_weights: Mapping[str, float]
    false_positive_weights: Mapping[str, float]
    teacher_fidelity: Mapping[str, Any]
    soft_dice_epsilon: float
    shapley: Mapping[str, Any]
    allocation: Mapping[str, Any]
    split_counts: Mapping[str, int]
    resume_enabled: bool
    dry_run_complete_compute: bool
    expected_plan_sha_required_for_official: bool
    primary_target_dtype: str
    repository_identity_gate_enabled: bool
    expected_contribution_contract_tag: str | None
    expected_contribution_contract_commit: str | None


def _config_int(value: Any, field: str) -> int:
    if not _is_int(value):
        _fail("B2_CONTRIBUTION_CONFIG_INVALID", f"{field} must be an integer")
    return int(value)


def _config_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("B2_CONTRIBUTION_CONFIG_INVALID", f"{field} must be a non-empty string")
    return value


def _config_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        _fail("B2_CONTRIBUTION_CONFIG_INVALID", f"{field} must be a boolean")
    return value


def _config_real(value: Any, field: str) -> float:
    if not _is_real(value):
        _fail("B2_CONTRIBUTION_CONFIG_INVALID", f"{field} must be a finite number")
    return float(value)


def _config_int_tuple(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, list | tuple) or not value:
        _fail("B2_CONTRIBUTION_CONFIG_INVALID", f"{field} must be a non-empty list")
    return tuple(_config_int(item, f"{field} entry") for item in value)


def _config_str_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple) or not value:
        _fail("B2_CONTRIBUTION_CONFIG_INVALID", f"{field} must be a non-empty list")
    return tuple(_config_str(item, f"{field} entry") for item in value)


def _config_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("B2_CONTRIBUTION_CONFIG_INVALID", f"{field} must be an object")
    return {str(key): item for key, item in value.items()}


def _config_optional_str(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _config_str(value, field)


def _numeric_mapping_equal(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    if set(actual) != set(expected):
        return False
    for key, reference in expected.items():
        observed = actual[key]
        if isinstance(reference, bool) or isinstance(observed, bool):
            if observed is not reference:
                return False
        elif isinstance(reference, str):
            if observed != reference:
                return False
        elif not _is_real(observed) or float(observed) != float(reference):
            return False
    return True


def _validate_pinned_contribution_config(config: ContributionTargetsConfig) -> None:
    tracked = config.configuration_id == TRACKED_CONFIGURATION_ID
    gate_c_profile = (
        config.contract_stage == CONTRACT_STAGE
        and config.repository_identity_gate_enabled is False
        and config.expected_contribution_contract_tag is None
        and config.expected_contribution_contract_commit is None
    )
    official_profile = (
        config.configuration_id == OFFICIAL_CONFIGURATION_ID
        and config.contract_stage == OFFICIAL_CONTRACT_STAGE
        and config.official_materialization_enabled is True
        and config.expected_input_artifact_kind == PRODUCTION_ARTIFACT_KIND
        and config.repository_identity_gate_enabled is True
        and config.expected_contribution_contract_tag == EXPECTED_CONTRIBUTION_CONTRACT_TAG
        and config.expected_contribution_contract_commit == EXPECTED_CONTRIBUTION_CONTRACT_COMMIT
    )
    if (
        config.schema_version != CONFIG_SCHEMA_VERSION
        or not (gate_c_profile or official_profile)
        or config.expected_input_artifact_kind not in _ACCEPTED_INPUT_ARTIFACT_KINDS
        or config.expected_execution_profile_sha256 != _EXPECTED_EXECUTION_PROFILE_SHA256
        or config.expected_split_scientific_sha256 != _EXPECTED_SPLIT_SCIENTIFIC_SHA256
        or config.expected_checkpoint_sha256 != _EXPECTED_CHECKPOINT_SHA256
        or config.expected_teacher_cache_scientific_sha256
        != _EXPECTED_TEACHER_CACHE_SCIENTIFIC_SHA256
        or config.expected_teacher_cache_sample_coverage_sha256
        != _EXPECTED_TEACHER_CACHE_SAMPLE_COVERAGE_SHA256
        or config.expected_descriptor_collection_scientific_sha256
        != _EXPECTED_DESCRIPTOR_COLLECTION_SCIENTIFIC_SHA256
        or config.expected_descriptor_sample_coverage_sha256
        != _EXPECTED_DESCRIPTOR_SAMPLE_COVERAGE_SHA256
        or config.expected_descriptor_normalization_scientific_sha256
        != _EXPECTED_DESCRIPTOR_NORMALIZATION_SCIENTIFIC_SHA256
        or config.expected_descriptor_normalization_training_coverage_sha256
        != _EXPECTED_DESCRIPTOR_NORMALIZATION_TRAINING_COVERAGE_SHA256
        or config.candidate_layers != DEFAULT_CANDIDATE_LAYERS
        or config.prediction_depths != DEFAULT_PREDICTION_DEPTHS
        or config.target_families != TARGET_FAMILIES
        or config.coalition_fusion != _EXPECTED_COALITION_FUSION
        or config.coalition_encoding != _EXPECTED_COALITION_ENCODING
        or not _numeric_mapping_equal(config.gt_calibration, _EXPECTED_GT_CALIBRATION)
        or not _numeric_mapping_equal(
            config.abnormal_gt_weights, _EXPECTED_ABNORMAL_GT_WEIGHTS
        )
        or not _numeric_mapping_equal(
            config.false_positive_weights, _EXPECTED_FALSE_POSITIVE_WEIGHTS
        )
        or not _numeric_mapping_equal(config.teacher_fidelity, _EXPECTED_TEACHER_FIDELITY)
        or float(config.soft_dice_epsilon) != SOFT_DICE_EPS
        or not _numeric_mapping_equal(config.shapley, _EXPECTED_SHAPLEY)
        or not _numeric_mapping_equal(config.allocation, _EXPECTED_ALLOCATION)
        or dict(config.split_counts) != dict(REQUIRED_SPLIT_COUNTS)
        or config.resume_enabled is not False
        or config.dry_run_complete_compute is not True
        or config.expected_plan_sha_required_for_official is not True
        or config.primary_target_dtype != STATISTICS_DTYPE
        or (tracked and config.official_materialization_enabled is not False)
        or (tracked and config.expected_input_artifact_kind != PRODUCTION_ARTIFACT_KIND)
    ):
        _fail("B2_CONTRIBUTION_CONFIG_DRIFT", "contribution-target Gate-C config drifted")
    return None


def load_contribution_targets_config(path: Any) -> ContributionTargetsConfig:
    """Load, type-check, and pin one contribution-target configuration file."""

    config_path = Path(path)
    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        _fail("B2_CONTRIBUTION_CONFIG_MISSING", f"path does not exist: {config_path}")
    except OSError as exc:
        _fail("B2_CONTRIBUTION_CONFIG_INVALID", f"cannot read {config_path}: {exc}")
    try:
        raw = json.loads(raw_text)
    except (json.JSONDecodeError, UnicodeError) as exc:
        _fail("B2_CONTRIBUTION_CONFIG_INVALID", f"invalid JSON at {config_path}: {exc}")
    if not isinstance(raw, Mapping):
        _fail("B2_CONTRIBUTION_CONFIG_INVALID", "config root must be an object")

    split_counts = {
        str(key): _config_int(value, f"split_counts.{key}")
        for key, value in _config_mapping(raw.get("split_counts"), "split_counts").items()
    }
    config = ContributionTargetsConfig(
        schema_version=_config_int(raw.get("schema_version"), "schema_version"),
        configuration_id=_config_str(raw.get("configuration_id"), "configuration_id"),
        contract_stage=_config_str(raw.get("contract_stage"), "contract_stage"),
        official_materialization_enabled=_config_bool(
            raw.get("official_materialization_enabled"), "official_materialization_enabled"
        ),
        expected_input_artifact_kind=_config_str(
            raw.get("expected_input_artifact_kind", PRODUCTION_ARTIFACT_KIND),
            "expected_input_artifact_kind",
        ),
        expected_execution_profile_sha256=_config_str(
            raw.get("expected_execution_profile_sha256"),
            "expected_execution_profile_sha256",
        ),
        expected_split_scientific_sha256=_config_str(
            raw.get("expected_split_scientific_sha256"), "expected_split_scientific_sha256"
        ),
        expected_checkpoint_sha256=_config_str(
            raw.get("expected_checkpoint_sha256"), "expected_checkpoint_sha256"
        ),
        expected_teacher_cache_scientific_sha256=_config_str(
            raw.get("expected_teacher_cache_scientific_sha256"),
            "expected_teacher_cache_scientific_sha256",
        ),
        expected_teacher_cache_sample_coverage_sha256=_config_str(
            raw.get("expected_teacher_cache_sample_coverage_sha256"),
            "expected_teacher_cache_sample_coverage_sha256",
        ),
        expected_descriptor_collection_scientific_sha256=_config_str(
            raw.get("expected_descriptor_collection_scientific_sha256"),
            "expected_descriptor_collection_scientific_sha256",
        ),
        expected_descriptor_sample_coverage_sha256=_config_str(
            raw.get("expected_descriptor_sample_coverage_sha256"),
            "expected_descriptor_sample_coverage_sha256",
        ),
        expected_descriptor_normalization_scientific_sha256=_config_str(
            raw.get("expected_descriptor_normalization_scientific_sha256"),
            "expected_descriptor_normalization_scientific_sha256",
        ),
        expected_descriptor_normalization_training_coverage_sha256=_config_str(
            raw.get("expected_descriptor_normalization_training_coverage_sha256"),
            "expected_descriptor_normalization_training_coverage_sha256",
        ),
        candidate_layers=_config_int_tuple(raw.get("candidate_layers"), "candidate_layers"),
        prediction_depths=_config_int_tuple(
            raw.get("prediction_depths"), "prediction_depths"
        ),
        target_families=_config_str_tuple(raw.get("target_families"), "target_families"),
        coalition_fusion=_config_str(raw.get("coalition_fusion"), "coalition_fusion"),
        coalition_encoding=_config_str(raw.get("coalition_encoding"), "coalition_encoding"),
        gt_calibration=MappingProxyType(
            _config_mapping(raw.get("gt_calibration"), "gt_calibration")
        ),
        abnormal_gt_weights=MappingProxyType(
            _config_mapping(raw.get("abnormal_gt_weights"), "abnormal_gt_weights")
        ),
        false_positive_weights=MappingProxyType(
            _config_mapping(raw.get("false_positive_weights"), "false_positive_weights")
        ),
        teacher_fidelity=MappingProxyType(
            _config_mapping(raw.get("teacher_fidelity"), "teacher_fidelity")
        ),
        soft_dice_epsilon=_config_real(raw.get("soft_dice_epsilon"), "soft_dice_epsilon"),
        shapley=MappingProxyType(_config_mapping(raw.get("shapley"), "shapley")),
        allocation=MappingProxyType(_config_mapping(raw.get("allocation"), "allocation")),
        split_counts=MappingProxyType(split_counts),
        resume_enabled=_config_bool(raw.get("resume_enabled"), "resume_enabled"),
        dry_run_complete_compute=_config_bool(
            raw.get("dry_run_complete_compute"), "dry_run_complete_compute"
        ),
        expected_plan_sha_required_for_official=_config_bool(
            raw.get("expected_plan_sha_required_for_official"),
            "expected_plan_sha_required_for_official",
        ),
        primary_target_dtype=_config_str(
            raw.get("primary_target_dtype"), "primary_target_dtype"
        ),
        repository_identity_gate_enabled=_config_bool(
            raw.get("repository_identity_gate_enabled"), "repository_identity_gate_enabled"
        ),
        expected_contribution_contract_tag=_config_optional_str(
            raw.get("expected_contribution_contract_tag"),
            "expected_contribution_contract_tag",
        ),
        expected_contribution_contract_commit=_config_optional_str(
            raw.get("expected_contribution_contract_commit"),
            "expected_contribution_contract_commit",
        ),
    )
    _validate_pinned_contribution_config(config)
    return config


# ---------------------------------------------------------------------------
# Shared input bundle and collection construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContributionInputSample:
    """One upstream-bound sample the shared collection path consumes."""

    stable_sample_id: str
    split_membership: str
    category: str
    label: int
    anomaly_type: str
    mask_identity: str | None
    maps_by_depth: Mapping[int, Mapping[int, Any]]
    mask: Any
    full_depth_map: Any
    teacher_record: Mapping[str, Any]
    teacher_record_scientific_sha256: str
    descriptor_record: Mapping[str, Any]


@dataclass(frozen=True)
class ContributionInputBundle:
    """The complete accepted upstream boundary for one contribution-target run."""

    artifact_kind: str
    candidate_layers: tuple[int, ...]
    prediction_depths: tuple[int, ...]
    samples: tuple[ContributionInputSample, ...]
    teacher_cache_scientific_sha256: str
    teacher_cache_sample_coverage_sha256: str
    descriptor_collection_scientific_sha256: str
    split_scientific_sha256: str
    checkpoint_sha256: str
    execution_profile_sha256: str


@dataclass(frozen=True)
class ContributionTargetCollection:
    """The complete in-memory scientific result of one contribution-target run."""

    artifact_kind: str
    candidate_layers: tuple[int, ...]
    prediction_depths: tuple[int, ...]
    calibration_artifact: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]
    normalization: Mapping[str, Any]
    plan: Mapping[str, Any]


def _effective_lattice(
    config: ContributionTargetsConfig,
    candidate_layers: Sequence[int] | None,
    prediction_depths: Sequence[int] | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    layers = _validate_candidate_layers(
        config.candidate_layers if candidate_layers is None else candidate_layers
    )
    depths = _validate_prediction_depths(
        config.prediction_depths if prediction_depths is None else prediction_depths
    )
    return layers, depths


def _validate_input_bundle(
    inputs: Any,
    *,
    candidate_layers: tuple[int, ...],
    prediction_depths: tuple[int, ...],
    split_counts: Mapping[str, int],
) -> ContributionInputBundle:
    if not isinstance(inputs, ContributionInputBundle):
        _fail("B2_CONTRIBUTION_INPUT_BUNDLE_INVALID", "inputs must be a ContributionInputBundle")
    if inputs.artifact_kind not in _ACCEPTED_INPUT_ARTIFACT_KINDS:
        _fail(
            "B2_TARGET_ARTIFACT_KIND_INVALID",
            f"artifact_kind {inputs.artifact_kind!r} is not accepted",
        )
    if (
        tuple(inputs.candidate_layers) != candidate_layers
        or tuple(inputs.prediction_depths) != prediction_depths
    ):
        _fail(
            "B2_CONTRIBUTION_INPUT_LATTICE_MISMATCH",
            "input bundle lattice differs from the requested candidate layers or depths",
        )
    observed: dict[str, int] = {membership: 0 for membership in SPLIT_MEMBERSHIPS}
    seen: set[str] = set()
    for sample in inputs.samples:
        if not isinstance(sample, ContributionInputSample):
            _fail(
                "B2_CONTRIBUTION_INPUT_BUNDLE_INVALID",
                "every input sample must be a ContributionInputSample",
            )
        if sample.split_membership not in observed:
            _fail(
                "B2_TARGET_RECORD_MEMBERSHIP_INVALID",
                f"split_membership {sample.split_membership!r} is not allowed",
            )
        if sample.stable_sample_id in seen:
            _fail(
                "B2_CONTRIBUTION_INPUT_DUPLICATE_SAMPLE",
                f"sample {sample.stable_sample_id} appears more than once",
            )
        seen.add(sample.stable_sample_id)
        observed[sample.split_membership] += 1
    if observed != dict(split_counts):
        _fail(
            "B2_TARGET_COVERAGE_COUNT_MISMATCH",
            f"input split counts {observed} do not match the configured {dict(split_counts)}",
        )
    return inputs


def _ordered_input_samples(
    inputs: ContributionInputBundle,
) -> tuple[ContributionInputSample, ...]:
    return tuple(sorted(inputs.samples, key=lambda sample: str(sample.stable_sample_id)))


def run_contribution_target_collection(
    *,
    config: ContributionTargetsConfig,
    inputs: Any,
    candidate_layers: Sequence[int] | None = None,
    prediction_depths: Sequence[int] | None = None,
) -> ContributionTargetCollection:
    """Compute the complete contribution-target collection purely in memory.

    This is the single shared construction path used by the dry run and by
    official materialization, so the two can never drift: source-training-only
    GT calibration, all dual-family sample records, the training-only Shapley
    normalization, the layered coverage identities, and the plan hash.
    """

    layers, depths = _effective_lattice(config, candidate_layers, prediction_depths)
    bundle = _validate_input_bundle(
        inputs,
        candidate_layers=layers,
        prediction_depths=depths,
        split_counts=config.split_counts,
    )
    ordered_samples = _ordered_input_samples(bundle)
    training_samples = tuple(
        sample for sample in ordered_samples if sample.split_membership == _TRAINING_MEMBERSHIP
    )

    calibration = fit_gt_map_calibration(
        tuple(
            GtCalibrationSample(
                stable_sample_id=sample.stable_sample_id,
                membership=sample.split_membership,
                maps_by_depth=sample.maps_by_depth,
            )
            for sample in training_samples
        ),
        candidate_layers=layers,
        prediction_depths=depths,
        quantiles=(
            float(config.gt_calibration["quantile_low"]),
            float(config.gt_calibration["quantile_high"]),
        ),
    )
    calibration_artifact = build_gt_map_calibration_artifact(
        calibration,
        source_teacher_record_scientific_sha256_by_id={
            sample.stable_sample_id: sample.teacher_record_scientific_sha256
            for sample in training_samples
        },
        teacher_cache_scientific_sha256=bundle.teacher_cache_scientific_sha256,
        teacher_cache_sample_coverage_sha256=bundle.teacher_cache_sample_coverage_sha256,
        descriptor_collection_scientific_sha256=(
            bundle.descriptor_collection_scientific_sha256
        ),
        split_scientific_sha256=bundle.split_scientific_sha256,
        checkpoint_sha256=bundle.checkpoint_sha256,
        execution_profile_sha256=bundle.execution_profile_sha256,
        expected_training_count=int(config.split_counts[_TRAINING_MEMBERSHIP]),
        artifact_kind=bundle.artifact_kind,
    )

    records: list[Mapping[str, Any]] = []
    for sample in ordered_samples:
        deepest_maps = sample.maps_by_depth[max(depths)]
        reconstructed = reconstruct_full_depth_teacher(
            deepest_maps, candidate_layers=layers
        )
        verify_full_depth_teacher_bitexact(reconstructed, sample.full_depth_map)
        upstream = bind_upstream_identities(
            teacher_record=sample.teacher_record,
            teacher_record_scientific_sha256=sample.teacher_record_scientific_sha256,
            teacher_cache_scientific_sha256=bundle.teacher_cache_scientific_sha256,
            teacher_cache_sample_coverage_sha256=(
                bundle.teacher_cache_sample_coverage_sha256
            ),
            descriptor_record=sample.descriptor_record,
            descriptor_collection_scientific_sha256=(
                bundle.descriptor_collection_scientific_sha256
            ),
            candidate_layers=layers,
            prediction_depths=depths,
        )
        is_anomalous = int(sample.label) == 1
        records.append(
            build_contribution_target_record(
                sample=ContributionTargetSample(
                    stable_sample_id=sample.stable_sample_id,
                    split_membership=sample.split_membership,
                    category=sample.category,
                    label=sample.label,
                    anomaly_type=sample.anomaly_type,
                    maps_by_depth=sample.maps_by_depth,
                    mask=sample.mask,
                    teacher_reference_map=sample.full_depth_map,
                ),
                calibration_artifact=calibration_artifact,
                upstream=upstream,
                mask_provenance=MaskProvenance(
                    mask_identity=sample.mask_identity,
                    mask_source=(
                        _MASK_SOURCE_ANOMALOUS if is_anomalous else _MASK_SOURCE_NORMAL
                    ),
                ),
                teacher_reference_provenance=TeacherReferenceProvenance(
                    cached_full_depth_map_digest=full_depth_map_digest(
                        sample.full_depth_map
                    ),
                    reconstruction_verified=True,
                    source_candidate_layers=layers,
                ),
                candidate_layers=layers,
                prediction_depths=depths,
                artifact_kind=bundle.artifact_kind,
            )
        )

    normalization = compute_shapley_normalization(
        [row for row in records if row["split_membership"] == _TRAINING_MEMBERSHIP],
        candidate_layers=layers,
        prediction_depths=depths,
        expected_training_count=int(config.split_counts[_TRAINING_MEMBERSHIP]),
        artifact_kind=bundle.artifact_kind,
    )
    plan = build_contribution_plan(
        records=records,
        calibration_artifact=calibration_artifact,
        normalization=normalization,
        candidate_layers=layers,
        prediction_depths=depths,
        official_materialization_enabled=config.official_materialization_enabled,
    )
    return ContributionTargetCollection(
        artifact_kind=bundle.artifact_kind,
        candidate_layers=layers,
        prediction_depths=depths,
        calibration_artifact=calibration_artifact,
        records=tuple(records),
        normalization=normalization,
        plan=plan,
    )


def coalition_counts_by_depth(
    candidate_layers: Sequence[int],
    prediction_depths: Sequence[int],
) -> dict[int, int]:
    """Total coalition count (including the empty coalition) per prediction depth."""

    layers = _validate_candidate_layers(candidate_layers)
    return {
        int(depth): 1 << len(players_for_depth(layers, depth))
        for depth in _validate_prediction_depths(prediction_depths)
    }


def dry_run_contribution_targets(
    *,
    config: ContributionTargetsConfig,
    inputs: Any,
    candidate_layers: Sequence[int] | None = None,
    prediction_depths: Sequence[int] | None = None,
    seed: int = 0,
    output_dir: Any = None,
) -> dict[str, Any]:
    """Compute the complete plan and report it without touching the filesystem.

    ``seed`` and ``output_dir`` are accepted for CLI parity and are reported
    back verbatim; neither participates in any scientific identity and
    ``output_dir`` is never created, probed, or locked.
    """

    collection = run_contribution_target_collection(
        config=config,
        inputs=inputs,
        candidate_layers=candidate_layers,
        prediction_depths=prediction_depths,
    )
    plan = collection.plan
    split_counts = dict(plan["planned_split_counts"])
    result: dict[str, Any] = {
        "mode": "dry_run",
        "status": "passed",
        "artifact_written": False,
        "run_directory_created": False,
        "teacher_forward_count": 0,
        "planned_samples": int(plan["planned_record_count"]),
        "training_targets": int(split_counts["training"]),
        "calibration_targets": int(split_counts["calibration"]),
        "evaluation_targets": int(split_counts["evaluation"]),
        "training_samples_for_gt_calibration": int(
            collection.calibration_artifact["training_sample_count"]
        ),
        "calibration_samples_for_gt_calibration": 0,
        "evaluation_samples_for_gt_calibration": 0,
        "training_samples_for_shapley_normalization": len(
            collection.normalization["ordered_training_stable_sample_ids"]
        ),
        "calibration_samples_for_shapley_normalization": 0,
        "evaluation_samples_for_shapley_normalization": 0,
        "prediction_depths": list(collection.prediction_depths),
        "candidate_layers": list(collection.candidate_layers),
        "coalition_counts": coalition_counts_by_depth(
            collection.candidate_layers, collection.prediction_depths
        ),
        "contribution_plan_scientific_sha256": str(
            plan["contribution_plan_scientific_sha256"]
        ),
        "official_materialization_enabled": bool(config.official_materialization_enabled),
        "configuration_id": config.configuration_id,
        "contract_stage": config.contract_stage,
        "artifact_kind": collection.artifact_kind,
        "planned_ordered_stable_sample_ids": list(
            plan["planned_ordered_stable_sample_ids"]
        ),
        "seed": int(seed),
        "output_dir": None if output_dir is None else str(output_dir),
    }
    for identity in SEVEN_LAYERED_IDENTITY_KEYS:
        result[identity] = str(plan[identity])
    return result


# ---------------------------------------------------------------------------
# Declared upstream roots
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_relative_path(relative_path: Any, *, code: str) -> str:
    if not isinstance(relative_path, str) or not relative_path:
        _fail(code, "path must be a non-empty relative path")
    raw = PurePosixPath(relative_path)
    if raw.is_absolute() or PurePath(relative_path).is_absolute():
        _fail(code, f"path must be relative: {relative_path}")
    parts = tuple(part for part in raw.parts if part != ".")
    if not parts or any(part in {"", ".."} for part in parts):
        _fail(code, f"path escapes the root or is malformed: {relative_path}")
    return PurePosixPath(*parts).as_posix()


def _resolve_within_root(
    root: Path,
    relative_path: str,
    *,
    expected_kind: str,
    path_code: str,
    missing_code: str,
    escape_code: str,
) -> Path:
    normalized = _normalized_relative_path(relative_path, code=path_code)
    resolved_root = Path(root).resolve()
    try:
        resolved = (resolved_root / normalized).resolve(strict=True)
    except OSError as exc:
        _fail(missing_code, f"{expected_kind} missing or unreadable at {normalized}: {exc}")
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        _fail(escape_code, f"{expected_kind} escapes the authoritative root: {normalized}")
    if not resolved.is_file():
        _fail(missing_code, f"{expected_kind} must be a regular file: {normalized}")
    return resolved


def _load_input_manifest(
    *,
    manifest_path: Path,
    root: Path,
    label: str,
) -> dict[str, Any]:
    resolved_root = Path(root).resolve()
    try:
        resolved_manifest = Path(manifest_path).resolve(strict=True)
    except OSError as exc:
        _fail(
            "B2_CONTRIBUTION_INPUT_MANIFEST_MISSING",
            f"{label} manifest missing or unreadable: {exc}",
        )
    try:
        resolved_manifest.relative_to(resolved_root)
    except ValueError:
        _fail(
            "B2_CONTRIBUTION_INPUT_MANIFEST_OUTSIDE_ROOT",
            f"{label} manifest must resolve inside its declared root",
        )
    try:
        payload = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail("B2_CONTRIBUTION_INPUT_MANIFEST_INVALID", f"invalid {label} manifest: {exc}")
    if not isinstance(payload, Mapping):
        _fail("B2_CONTRIBUTION_INPUT_MANIFEST_INVALID", f"{label} manifest must be an object")
    if payload.get("status") != "passed":
        _fail(
            "B2_CONTRIBUTION_INPUT_MANIFEST_NOT_PASSED",
            f"{label} manifest status must be 'passed'",
        )
    kind = payload.get("artifact_kind")
    if kind not in _ACCEPTED_INPUT_ARTIFACT_KINDS:
        _fail(
            "B2_TARGET_ARTIFACT_KIND_INVALID",
            f"{label} manifest artifact_kind {kind!r} is not accepted",
        )
    return dict(payload)


def _load_pt_payload(path: Path, *, code: str) -> Mapping[str, Any]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - any unpickling failure is fail-closed
        _fail(code, f"cannot load artifact payload at {path.name}: {exc}")
    if not isinstance(loaded, Mapping):
        _fail(code, f"artifact payload at {path.name} must be a mapping")
    return loaded


def load_contribution_inputs_from_disk(
    *,
    config: ContributionTargetsConfig,
    teacher_cache_manifest_path: Any,
    teacher_cache_root: Any,
    descriptor_manifest_path: Any,
    descriptor_root: Any,
) -> ContributionInputBundle:
    """Read a declared teacher-cache and descriptor collection into one bundle.

    Production inputs are bound to the configured upstream identities. Hermetic
    ``test_fixture`` inputs skip that binding but may never reach official
    materialization unless the configuration explicitly declares fixture inputs,
    which the tracked Gate-C configuration can never do.
    """

    teacher_root = Path(teacher_cache_root)
    descriptor_root_path = Path(descriptor_root)
    teacher_manifest = _load_input_manifest(
        manifest_path=Path(teacher_cache_manifest_path),
        root=teacher_root,
        label="teacher-cache",
    )
    descriptor_manifest = _load_input_manifest(
        manifest_path=Path(descriptor_manifest_path),
        root=descriptor_root_path,
        label="descriptor",
    )
    artifact_kind = str(teacher_manifest["artifact_kind"])
    if descriptor_manifest["artifact_kind"] != artifact_kind:
        _fail(
            "B2_TARGET_ARTIFACT_KIND_INVALID",
            "teacher-cache and descriptor manifests declare different artifact kinds",
        )
    if artifact_kind == TEST_FIXTURE_ARTIFACT_KIND and (
        config.official_materialization_enabled
        and config.expected_input_artifact_kind != TEST_FIXTURE_ARTIFACT_KIND
    ):
        _fail(
            "B2_TARGET_TEST_FIXTURE_NOT_ACCEPTED",
            "test_fixture artifacts are never accepted by production official mode",
        )

    teacher_cache_hash = str(teacher_manifest.get("cache_scientific_sha256", ""))
    teacher_coverage_hash = str(teacher_manifest.get("sample_coverage_sha256", ""))
    descriptor_collection_hash = str(
        descriptor_manifest.get("descriptor_collection_scientific_sha256", "")
    )
    split_hash = str(teacher_manifest.get("split_scientific_sha256", ""))
    checkpoint_hash = str(teacher_manifest.get("checkpoint_sha256", ""))
    profile_hash = str(teacher_manifest.get("execution_profile_sha256", ""))
    for digest, label in (
        (teacher_cache_hash, "teacher cache"),
        (teacher_coverage_hash, "teacher cache coverage"),
        (descriptor_collection_hash, "descriptor collection"),
        (split_hash, "split manifest"),
        (checkpoint_hash, "checkpoint"),
        (profile_hash, "execution profile"),
    ):
        if not _is_sha256(digest):
            _fail("B2_TARGET_UPSTREAM_HASH_INVALID", f"{label} hash is invalid")
    if artifact_kind == PRODUCTION_ARTIFACT_KIND and (
        teacher_cache_hash != config.expected_teacher_cache_scientific_sha256
        or teacher_coverage_hash != config.expected_teacher_cache_sample_coverage_sha256
        or descriptor_collection_hash
        != config.expected_descriptor_collection_scientific_sha256
        or split_hash != config.expected_split_scientific_sha256
        or checkpoint_hash != config.expected_checkpoint_sha256
        or profile_hash != config.expected_execution_profile_sha256
    ):
        _fail(
            "B2_CONTRIBUTION_UPSTREAM_IDENTITY_MISMATCH",
            "declared upstream identities drifted from the configured expectations",
        )

    descriptor_rows = descriptor_manifest.get("samples")
    teacher_rows = teacher_manifest.get("samples")
    if not isinstance(teacher_rows, Sequence) or not isinstance(descriptor_rows, Sequence):
        _fail("B2_CONTRIBUTION_INPUT_MANIFEST_INVALID", "manifest samples must be lists")
    descriptor_by_id = {str(row["stable_sample_id"]): row for row in descriptor_rows}

    samples: list[ContributionInputSample] = []
    for row in teacher_rows:
        stable_sample_id = str(row["stable_sample_id"])
        record_path = _resolve_within_root(
            teacher_root,
            str(row["relative_path"]),
            expected_kind=f"teacher record {stable_sample_id}",
            path_code="B2_CONTRIBUTION_INPUT_RELATIVE_PATH_INVALID",
            missing_code="B2_CONTRIBUTION_MISSING_ARTIFACT",
            escape_code="B2_CONTRIBUTION_INPUT_ROOT_ESCAPE",
        )
        if _sha256_file(record_path) != str(row["record_file_sha256"]):
            _fail(
                "B2_CONTRIBUTION_RECORD_FILE_HASH_MISMATCH",
                f"teacher record file hash mismatch for {stable_sample_id}",
            )
        payload = _load_pt_payload(
            record_path, code="B2_CONTRIBUTION_INPUT_PAYLOAD_INVALID"
        )
        teacher_record = payload["scientific_record"]
        maps_by_depth = {
            int(depth_key): {
                int(layer_key): tensor for layer_key, tensor in layer_maps.items()
            }
            for depth_key, layer_maps in payload["maps_by_depth"].items()
        }
        descriptor_row = descriptor_by_id.get(stable_sample_id)
        if descriptor_row is None:
            _fail(
                "B2_TARGET_UPSTREAM_SAMPLE_MISMATCH",
                f"descriptor collection has no record for {stable_sample_id}",
            )
        descriptor_path = _resolve_within_root(
            descriptor_root_path,
            str(descriptor_row["relative_record_path"]),
            expected_kind=f"descriptor record {stable_sample_id}",
            path_code="B2_CONTRIBUTION_INPUT_RELATIVE_PATH_INVALID",
            missing_code="B2_CONTRIBUTION_MISSING_ARTIFACT",
            escape_code="B2_CONTRIBUTION_INPUT_ROOT_ESCAPE",
        )
        descriptor_payload = _load_pt_payload(
            descriptor_path, code="B2_CONTRIBUTION_INPUT_PAYLOAD_INVALID"
        )
        samples.append(
            ContributionInputSample(
                stable_sample_id=stable_sample_id,
                split_membership=str(row["membership"]),
                category=str(teacher_record["category"]),
                label=int(teacher_record["image_label"]),
                anomaly_type=str(teacher_record["anomaly_type"]),
                mask_identity=teacher_record.get("mask_identity"),
                maps_by_depth=maps_by_depth,
                mask=payload["mask"],
                full_depth_map=payload["full_depth_map"],
                teacher_record=teacher_record,
                teacher_record_scientific_sha256=str(row["record_scientific_sha256"]),
                descriptor_record=descriptor_payload["scientific_record"],
            )
        )

    return ContributionInputBundle(
        artifact_kind=artifact_kind,
        candidate_layers=tuple(int(layer) for layer in teacher_manifest["candidate_layers"]),
        prediction_depths=tuple(
            int(depth) for depth in teacher_manifest["prediction_depths"]
        ),
        samples=tuple(samples),
        teacher_cache_scientific_sha256=teacher_cache_hash,
        teacher_cache_sample_coverage_sha256=teacher_coverage_hash,
        descriptor_collection_scientific_sha256=descriptor_collection_hash,
        split_scientific_sha256=split_hash,
        checkpoint_sha256=checkpoint_hash,
        execution_profile_sha256=profile_hash,
    )


def dry_run_contribution_targets_from_roots(
    *,
    config: ContributionTargetsConfig,
    teacher_cache_manifest_path: Any,
    teacher_cache_root: Any,
    descriptor_manifest_path: Any,
    descriptor_root: Any,
    seed: int = 0,
    output_dir: Any = None,
) -> dict[str, Any]:
    """Read declared upstream roots and produce the complete no-write dry run."""

    inputs = load_contribution_inputs_from_disk(
        config=config,
        teacher_cache_manifest_path=teacher_cache_manifest_path,
        teacher_cache_root=teacher_cache_root,
        descriptor_manifest_path=descriptor_manifest_path,
        descriptor_root=descriptor_root,
    )
    return dry_run_contribution_targets(
        config=config, inputs=inputs, seed=seed, output_dir=output_dir
    )


def dry_run_contribution_targets_from_fixture_roots(
    *,
    config: ContributionTargetsConfig,
    teacher_cache_manifest_path: Any,
    teacher_cache_root: Any,
    descriptor_manifest_path: Any,
    descriptor_root: Any,
    seed: int = 0,
    output_dir: Any = None,
) -> dict[str, Any]:
    """Hermetic dry run over ``test_fixture`` upstream roots."""

    inputs = load_contribution_inputs_from_disk(
        config=config,
        teacher_cache_manifest_path=teacher_cache_manifest_path,
        teacher_cache_root=teacher_cache_root,
        descriptor_manifest_path=descriptor_manifest_path,
        descriptor_root=descriptor_root,
    )
    if inputs.artifact_kind != TEST_FIXTURE_ARTIFACT_KIND:
        _fail(
            "B2_TARGET_ARTIFACT_KIND_INVALID",
            "the fixture dry-run entry point only accepts test_fixture inputs",
        )
    return dry_run_contribution_targets(
        config=config, inputs=inputs, seed=seed, output_dir=output_dir
    )


# ---------------------------------------------------------------------------
# Atomic dual-hash persistence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PersistedTargetRecordEntry:
    """One verified on-disk contribution-target record."""

    stable_sample_id: str
    relative_record_path: str
    contribution_target_record_scientific_sha256: str
    contribution_target_record_file_sha256: str
    verification_status: str = "verified"


@dataclass(frozen=True)
class PersistedStatisticsEntry:
    """One verified on-disk calibration or normalization artifact."""

    relative_path: str
    scientific_sha256: str
    file_sha256: str
    verification_status: str = "verified"


@dataclass(frozen=True)
class ContributionTargetMaterializationResult:
    """The outcome of one official contribution-target materialization."""

    run_dir: Path
    manifest: Mapping[str, Any]
    contribution_plan_scientific_sha256: str
    teacher_forward_count: int


@dataclass(frozen=True)
class ContributionRepositoryIdentity:
    """Verified outer Git provenance for one official materialization."""

    contract_tag: str
    contract_commit: str
    generation_commit: str
    head_is_descendant: bool
    worktree_clean: bool


@dataclass(frozen=True)
class VerifiedContributionTargetCollection:
    """A disk-verified contribution-target collection."""

    run_dir: Path
    manifest: Mapping[str, Any]
    records_by_id: Mapping[str, Mapping[str, Any]]
    calibration_artifact: Mapping[str, Any]
    normalization: Mapping[str, Any]
    teacher_forward_count: int


# Verification is an instance-identity property, never a copyable field: only the
# exact objects returned by ``verify_contribution_target_collection`` after every
# check are registered here, so manual construction and ``dataclasses.replace``
# stay unverified.
_VERIFIED_COLLECTION_REGISTRY: dict[
    int, weakref.ReferenceType[VerifiedContributionTargetCollection]
] = {}


def _seal_verified_collection(
    collection: VerifiedContributionTargetCollection,
) -> VerifiedContributionTargetCollection:
    key = id(collection)

    def _forget(_reference: Any, key: int = key) -> None:
        _VERIFIED_COLLECTION_REGISTRY.pop(key, None)

    _VERIFIED_COLLECTION_REGISTRY[key] = weakref.ref(collection, _forget)
    return collection


def _is_verified_collection(value: Any) -> bool:
    if not isinstance(value, VerifiedContributionTargetCollection):
        return False
    reference = _VERIFIED_COLLECTION_REGISTRY.get(id(value))
    return reference is not None and reference() is value


@dataclass(frozen=True)
class ContributionTargetCollectionComparison:
    """Exact scientific and byte-level comparison of two verified collections."""

    scientifically_equivalent: bool
    reasons: tuple[str, ...]
    layered_identities_equal: bool
    record_scientific_hashes_equal: bool
    utility_tables_equal: bool
    signed_shapley_equal: bool
    allocations_equal: bool
    gt_calibration_equal: bool
    shapley_normalization_equal: bool
    coverage_equal: bool
    teacher_forward_count_equal: bool
    file_byte_equal: bool


def _run_contribution_git(repository_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        _fail("B2_CONTRIBUTION_REPOSITORY_INVALID", f"cannot execute Git: {exc}")


def verify_contribution_repository_identity(
    *,
    config: ContributionTargetsConfig,
    repository_root: Path,
    expected_generation_commit: str | None = None,
) -> ContributionRepositoryIdentity:
    """Verify the frozen contract tag, descendant HEAD, and clean worktree."""

    root = Path(repository_root)
    if (
        not config.repository_identity_gate_enabled
        or config.expected_contribution_contract_tag != EXPECTED_CONTRIBUTION_CONTRACT_TAG
        or config.expected_contribution_contract_commit != EXPECTED_CONTRIBUTION_CONTRACT_COMMIT
    ):
        _fail(
            "B2_CONTRIBUTION_REPOSITORY_IDENTITY_CONFIG_INVALID",
            "repository identity verification requires the frozen B2-04B contract",
        )
    tag_result = _run_contribution_git(
        root, "rev-parse", "--verify", f"{config.expected_contribution_contract_tag}^{{commit}}"
    )
    resolved_tag = tag_result.stdout.strip()
    if tag_result.returncode != 0 or resolved_tag != config.expected_contribution_contract_commit:
        _fail(
            "B2_CONTRIBUTION_CONTRACT_TAG_INVALID",
            "the configured contribution contract tag is missing or moved",
        )
    head_result = _run_contribution_git(root, "rev-parse", "--verify", "HEAD")
    generation_commit = head_result.stdout.strip()
    if head_result.returncode != 0 or len(generation_commit) != 40:
        _fail("B2_CONTRIBUTION_REPOSITORY_INVALID", "cannot resolve repository HEAD")
    ancestor_result = _run_contribution_git(
        root,
        "merge-base",
        "--is-ancestor",
        config.expected_contribution_contract_commit,
        generation_commit,
    )
    if ancestor_result.returncode != 0:
        _fail(
            "B2_CONTRIBUTION_HEAD_NOT_DESCENDANT",
            "repository HEAD is not a descendant of the contribution contract commit",
        )
    if (
        expected_generation_commit is not None
        and generation_commit != expected_generation_commit
    ):
        _fail(
            "B2_CONTRIBUTION_GENERATION_COMMIT_CHANGED",
            "repository HEAD changed during contribution-target calculation",
        )
    status_result = _run_contribution_git(
        root, "status", "--porcelain", "--untracked-files=all"
    )
    if status_result.returncode != 0:
        _fail("B2_CONTRIBUTION_REPOSITORY_INVALID", "cannot inspect repository status")
    if status_result.stdout:
        _fail(
            "B2_CONTRIBUTION_WORKTREE_DIRTY",
            "official contribution-target materialization requires a clean worktree",
        )
    return ContributionRepositoryIdentity(
        contract_tag=config.expected_contribution_contract_tag,
        contract_commit=config.expected_contribution_contract_commit,
        generation_commit=generation_commit,
        head_is_descendant=True,
        worktree_clean=True,
    )


def contribution_record_relative_path(stable_sample_id: Any) -> str:
    """Run-relative path of one contribution-target record artifact."""

    if not _is_sha256(stable_sample_id):
        _fail("B2_CONTRIBUTION_SAMPLE_ID_INVALID", "stable_sample_id must be sha256 hex")
    return f"{RECORDS_DIRECTORY}/{stable_sample_id}.pt"


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle_fd, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _refuse_overwrite(path: Path) -> None:
    if path.exists() or path.is_symlink():
        _fail("B2_CONTRIBUTION_OVERWRITE_FORBIDDEN", f"refusing to overwrite {path.name}")


def write_contribution_target_record_atomic(
    destination: Any,
    record: Mapping[str, Any],
    *,
    candidate_layers: Sequence[int],
    prediction_depths: Sequence[int],
) -> PersistedTargetRecordEntry:
    """Persist one record atomically, then reload, rehash, and revalidate it."""

    path = Path(destination)
    stable_sample_id = str(record["stable_sample_id"])
    if path.name != f"{stable_sample_id}.pt":
        _fail(
            "B2_CONTRIBUTION_SAMPLE_ID_INVALID",
            "destination filename must equal the stable sample ID",
        )
    _refuse_overwrite(path)
    claimed = str(record["contribution_target_record_scientific_sha256"])
    if claimed != contribution_target_record_scientific_sha256(record):
        _fail(
            "B2_TARGET_RECORD_HASH_MISMATCH",
            "contribution_target_record_scientific_sha256 does not match content before write",
        )
    persistable = {
        key: value for key, value in dict(record).items() if key != _RECORD_FILE_HASH_FIELD
    }
    _atomic_torch_save(
        path,
        {
            "scientific_record": persistable,
            "contribution_target_record_scientific_sha256": claimed,
        },
    )
    file_digest = _sha256_file(path)
    entry = PersistedTargetRecordEntry(
        stable_sample_id=stable_sample_id,
        relative_record_path=contribution_record_relative_path(stable_sample_id),
        contribution_target_record_scientific_sha256=claimed,
        contribution_target_record_file_sha256=file_digest,
        verification_status="unverified",
    )
    _verify_record_payload(
        path,
        entry=entry,
        candidate_layers=candidate_layers,
        prediction_depths=prediction_depths,
    )
    return PersistedTargetRecordEntry(
        stable_sample_id=stable_sample_id,
        relative_record_path=entry.relative_record_path,
        contribution_target_record_scientific_sha256=claimed,
        contribution_target_record_file_sha256=file_digest,
        verification_status="verified",
    )


def _verify_record_payload(
    path: Path,
    *,
    entry: PersistedTargetRecordEntry,
    candidate_layers: Sequence[int],
    prediction_depths: Sequence[int],
) -> Mapping[str, Any]:
    if _sha256_file(path) != entry.contribution_target_record_file_sha256:
        _fail(
            "B2_CONTRIBUTION_RECORD_FILE_HASH_MISMATCH",
            f"record file hash mismatch for {entry.stable_sample_id}",
        )
    loaded = _load_pt_payload(path, code="B2_CONTRIBUTION_PT_PAYLOAD_INVALID")
    if set(loaded) != set(RECORD_PAYLOAD_KEYS):
        _fail("B2_CONTRIBUTION_PT_PAYLOAD_INVALID", "record payload keys are not exact")
    scientific_record = loaded["scientific_record"]
    if not isinstance(scientific_record, Mapping):
        _fail("B2_CONTRIBUTION_PT_PAYLOAD_INVALID", "scientific_record must be a mapping")
    if _RECORD_FILE_HASH_FIELD in scientific_record:
        _fail(
            "B2_CONTRIBUTION_FILE_HASH_IN_PAYLOAD",
            "the record file hash must never live inside the persisted payload",
        )
    if "depth_targets" not in scientific_record:
        _fail(
            "B2_CONTRIBUTION_DIGEST_ONLY_RECORD",
            f"record payload for {entry.stable_sample_id} carries no depth targets",
        )
    recomputed = contribution_target_record_scientific_sha256(scientific_record)
    if (
        loaded["contribution_target_record_scientific_sha256"]
        != entry.contribution_target_record_scientific_sha256
        or recomputed != entry.contribution_target_record_scientific_sha256
    ):
        _fail(
            "B2_TARGET_RECORD_HASH_MISMATCH",
            f"record scientific hash drifted for {entry.stable_sample_id}",
        )
    validate_contribution_target_record(
        scientific_record,
        candidate_layers=candidate_layers,
        prediction_depths=prediction_depths,
    )
    return scientific_record


def verify_persisted_contribution_target_record(
    *,
    run_dir: Any,
    entry: PersistedTargetRecordEntry,
    candidate_layers: Sequence[int],
    prediction_depths: Sequence[int],
) -> Mapping[str, Any]:
    """Re-verify one persisted record from disk against its manifest entry."""

    normalized = _normalized_relative_path(
        entry.relative_record_path, code="B2_CONTRIBUTION_RUN_RELATIVE_PATH_INVALID"
    )
    if normalized != contribution_record_relative_path(entry.stable_sample_id):
        _fail(
            "B2_CONTRIBUTION_RUN_RELATIVE_PATH_INVALID",
            f"record path drifted for {entry.stable_sample_id}",
        )
    path = _resolve_within_root(
        Path(run_dir),
        normalized,
        expected_kind=f"contribution-target record {entry.stable_sample_id}",
        path_code="B2_CONTRIBUTION_RUN_RELATIVE_PATH_INVALID",
        missing_code="B2_CONTRIBUTION_MISSING_ARTIFACT",
        escape_code="B2_CONTRIBUTION_RUN_ROOT_ESCAPE",
    )
    return _verify_record_payload(
        path,
        entry=entry,
        candidate_layers=candidate_layers,
        prediction_depths=prediction_depths,
    )


def write_gt_map_calibration_atomic(
    destination: Any,
    artifact: Mapping[str, Any],
) -> PersistedStatisticsEntry:
    """Persist the GT map calibration artifact atomically and re-verify it."""

    path = Path(destination)
    _refuse_overwrite(path)
    validate_gt_map_calibration_artifact(artifact)
    digest = str(artifact["gt_map_calibration_scientific_sha256"])
    _atomic_torch_save(
        path,
        {
            "scientific_calibration_record": dict(artifact),
            "gt_map_calibration_scientific_sha256": digest,
        },
    )
    entry = PersistedStatisticsEntry(
        relative_path=CALIBRATION_RELATIVE_PATH,
        scientific_sha256=digest,
        file_sha256=_sha256_file(path),
        verification_status="unverified",
    )
    _verify_calibration_payload(path, entry=entry)
    return PersistedStatisticsEntry(
        relative_path=entry.relative_path,
        scientific_sha256=entry.scientific_sha256,
        file_sha256=entry.file_sha256,
        verification_status="verified",
    )


def _verify_calibration_payload(
    path: Path, *, entry: PersistedStatisticsEntry
) -> Mapping[str, Any]:
    if _sha256_file(path) != entry.file_sha256:
        _fail(
            "B2_CONTRIBUTION_CALIBRATION_FILE_HASH_MISMATCH",
            "GT map calibration file hash mismatch",
        )
    loaded = _load_pt_payload(path, code="B2_CONTRIBUTION_PT_PAYLOAD_INVALID")
    if set(loaded) != set(CALIBRATION_PAYLOAD_KEYS):
        _fail("B2_CONTRIBUTION_PT_PAYLOAD_INVALID", "calibration payload keys are not exact")
    scientific = loaded["scientific_calibration_record"]
    if not isinstance(scientific, Mapping):
        _fail("B2_CONTRIBUTION_PT_PAYLOAD_INVALID", "calibration record must be a mapping")
    validate_gt_map_calibration_artifact(scientific)
    if (
        loaded["gt_map_calibration_scientific_sha256"] != entry.scientific_sha256
        or scientific["gt_map_calibration_scientific_sha256"] != entry.scientific_sha256
    ):
        _fail(
            "B2_TARGET_CALIBRATION_HASH_MISMATCH",
            "GT map calibration scientific hash drifted after reload",
        )
    return scientific


def write_shapley_normalization_atomic(
    destination: Any,
    artifact: Mapping[str, Any],
) -> PersistedStatisticsEntry:
    """Persist the Shapley normalization artifact atomically and re-verify it."""

    path = Path(destination)
    _refuse_overwrite(path)
    digest = str(artifact["shapley_normalization_scientific_sha256"])
    if digest != shapley_normalization_scientific_sha256(artifact):
        _fail(
            "B2_TARGET_NORMALIZATION_HASH_MISMATCH",
            "shapley_normalization_scientific_sha256 does not match content before write",
        )
    _atomic_torch_save(
        path,
        {
            "scientific_normalization_record": dict(artifact),
            "shapley_normalization_scientific_sha256": digest,
        },
    )
    entry = PersistedStatisticsEntry(
        relative_path=NORMALIZATION_RELATIVE_PATH,
        scientific_sha256=digest,
        file_sha256=_sha256_file(path),
        verification_status="unverified",
    )
    _verify_normalization_payload(path, entry=entry)
    return PersistedStatisticsEntry(
        relative_path=entry.relative_path,
        scientific_sha256=entry.scientific_sha256,
        file_sha256=entry.file_sha256,
        verification_status="verified",
    )


def _verify_normalization_payload(
    path: Path, *, entry: PersistedStatisticsEntry
) -> Mapping[str, Any]:
    if _sha256_file(path) != entry.file_sha256:
        _fail(
            "B2_CONTRIBUTION_NORMALIZATION_FILE_HASH_MISMATCH",
            "Shapley normalization file hash mismatch",
        )
    loaded = _load_pt_payload(path, code="B2_CONTRIBUTION_PT_PAYLOAD_INVALID")
    if set(loaded) != set(NORMALIZATION_PAYLOAD_KEYS):
        _fail(
            "B2_CONTRIBUTION_PT_PAYLOAD_INVALID", "normalization payload keys are not exact"
        )
    scientific = loaded["scientific_normalization_record"]
    if not isinstance(scientific, Mapping):
        _fail("B2_CONTRIBUTION_PT_PAYLOAD_INVALID", "normalization record must be a mapping")
    recomputed = shapley_normalization_scientific_sha256(scientific)
    if (
        loaded["shapley_normalization_scientific_sha256"] != entry.scientific_sha256
        or recomputed != entry.scientific_sha256
    ):
        _fail(
            "B2_TARGET_NORMALIZATION_HASH_MISMATCH",
            "Shapley normalization scientific hash drifted after reload",
        )
    return scientific


def write_final_manifest_with_receipt_atomic(
    run_dir: Any,
    manifest: Mapping[str, Any],
) -> str:
    """Atomically write ``final_manifest.json`` and its non-self-referential receipt."""

    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / FINAL_MANIFEST_NAME
    receipt_path = root / FINAL_MANIFEST_RECEIPT_NAME
    _refuse_overwrite(manifest_path)
    _refuse_overwrite(receipt_path)
    payload = {
        key: value
        for key, value in dict(manifest).items()
        if key != "final_manifest_file_sha256"
    }
    atomic_write_json(manifest_path, payload)
    digest = _sha256_file(manifest_path)
    handle_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{receipt_path.name}.", suffix=".tmp", dir=root
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            handle.write(digest + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, receipt_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return digest


def verify_final_manifest_receipt(run_dir: Any) -> str:
    """Require the receipt to match the manifest bytes exactly."""

    root = Path(run_dir)
    manifest_path = _resolve_within_root(
        root,
        FINAL_MANIFEST_NAME,
        expected_kind="final manifest",
        path_code="B2_CONTRIBUTION_RUN_RELATIVE_PATH_INVALID",
        missing_code="B2_CONTRIBUTION_MISSING_ARTIFACT",
        escape_code="B2_CONTRIBUTION_RUN_ROOT_ESCAPE",
    )
    receipt_path = _resolve_within_root(
        root,
        FINAL_MANIFEST_RECEIPT_NAME,
        expected_kind="final manifest receipt",
        path_code="B2_CONTRIBUTION_RUN_RELATIVE_PATH_INVALID",
        missing_code="B2_CONTRIBUTION_MISSING_ARTIFACT",
        escape_code="B2_CONTRIBUTION_RUN_ROOT_ESCAPE",
    )
    actual = _sha256_file(manifest_path)
    claimed = receipt_path.read_text(encoding="utf-8").strip()
    if claimed != actual:
        _fail(
            "B2_CONTRIBUTION_MANIFEST_RECEIPT_MISMATCH",
            "final_manifest.json.sha256 does not match the manifest bytes",
        )
    return actual


def _is_temporary_artifact(path: Path) -> bool:
    name = path.name
    return name.endswith(".tmp") or (name.startswith(".") and name != ".gitkeep")


def audit_contribution_target_integrity(
    *,
    run_dir: Any,
    manifest: Mapping[str, Any],
    planned_ids: Sequence[str],
) -> None:
    """Structural fail-closed audit of one run directory against its plan."""

    root = Path(run_dir)
    planned = [str(item) for item in planned_ids]
    planned_set = set(planned)
    if len(planned_set) != len(planned):
        _fail("B2_CONTRIBUTION_PLAN_DUPLICATE", "planned sample IDs contain duplicates")

    for path in sorted(root.rglob("*")):
        if path.is_file() and _is_temporary_artifact(path):
            _fail(
                "B2_CONTRIBUTION_TEMP_ARTIFACT_PRESENT",
                f"temporary artifact left behind: {path.name}",
            )

    records_dir = root / RECORDS_DIRECTORY
    on_disk: dict[str, Path] = {}
    if records_dir.is_dir():
        for path in sorted(records_dir.iterdir()):
            if path.suffix != ".pt" or not path.is_file():
                _fail(
                    "B2_CONTRIBUTION_ORPHAN_ARTIFACT",
                    f"unexpected entry in the records directory: {path.name}",
                )
            if path.stem not in planned_set:
                _fail(
                    "B2_CONTRIBUTION_ORPHAN_ARTIFACT",
                    f"orphan contribution-target record: {path.name}",
                )
            on_disk[path.stem] = path

    missing = [stable_id for stable_id in planned if stable_id not in on_disk]
    if missing:
        if manifest.get("status") == "passed":
            _fail(
                "B2_CONTRIBUTION_PARTIAL_CLAIMING_PASSED",
                f"manifest claims passed while {len(missing)} records are missing",
            )
        _fail(
            "B2_CONTRIBUTION_MISSING_ARTIFACT",
            f"missing contribution-target records: {missing[:3]}",
        )
    return None


def _statistics_entry_from_manifest(
    manifest: Mapping[str, Any],
    key: str,
    *,
    expected_relative_path: str,
) -> PersistedStatisticsEntry:
    block = manifest.get(key)
    if not isinstance(block, Mapping):
        _fail(
            "B2_CONTRIBUTION_MANIFEST_INVALID",
            f"final manifest is missing the {key} block",
        )
    return PersistedStatisticsEntry(
        relative_path=str(block.get("relative_path", expected_relative_path)),
        scientific_sha256=str(block["scientific_sha256"]),
        file_sha256=str(block["file_sha256"]),
        verification_status=str(block.get("verification_status", "verified")),
    )


def build_contribution_targets_manifest(
    *,
    config: ContributionTargetsConfig,
    collection: ContributionTargetCollection,
    record_entries: Sequence[PersistedTargetRecordEntry],
    calibration_entry: PersistedStatisticsEntry | None,
    normalization_entry: PersistedStatisticsEntry | None,
    repository_identity: ContributionRepositoryIdentity | None = None,
) -> dict[str, Any]:
    """Build the passed final manifest from dual-hash verified disk entries."""

    ordered_records = sorted(
        collection.records, key=lambda row: str(row["stable_sample_id"])
    )
    by_id = {entry.stable_sample_id: entry for entry in record_entries}
    if (
        calibration_entry is None
        or normalization_entry is None
        or len(record_entries) != len(ordered_records)
        or set(by_id) != {str(row["stable_sample_id"]) for row in ordered_records}
    ):
        _fail(
            "B2_CONTRIBUTION_PASSED_MANIFEST_REQUIRES_VERIFIED_RECORDS",
            "a passed manifest requires one verified entry per record plus both statistics",
        )
    for entry in record_entries:
        if (
            entry.verification_status != "verified"
            or not _is_sha256(entry.contribution_target_record_scientific_sha256)
            or not _is_sha256(entry.contribution_target_record_file_sha256)
        ):
            _fail(
                "B2_CONTRIBUTION_PASSED_MANIFEST_REQUIRES_VERIFIED_RECORDS",
                "every record entry must be dual-hash verified",
            )
    for statistics in (calibration_entry, normalization_entry):
        if (
            statistics.verification_status != "verified"
            or not _is_sha256(statistics.scientific_sha256)
            or not _is_sha256(statistics.file_sha256)
        ):
            _fail(
                "B2_CONTRIBUTION_PASSED_MANIFEST_REQUIRES_VERIFIED_RECORDS",
                "both statistics artifacts must be dual-hash verified",
            )

    plan = collection.plan
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "passed",
        "configuration_id": config.configuration_id,
        "contract_stage": config.contract_stage,
        "artifact_kind": collection.artifact_kind,
        "candidate_layers": list(collection.candidate_layers),
        "prediction_depths": list(collection.prediction_depths),
        "target_families": list(TARGET_FAMILIES),
        "split_counts": dict(plan["planned_split_counts"]),
        "planned_ordered_stable_sample_ids": list(
            plan["planned_ordered_stable_sample_ids"]
        ),
        "contribution_plan_scientific_sha256": str(
            plan["contribution_plan_scientific_sha256"]
        ),
        "official_materialization_enabled": bool(config.official_materialization_enabled),
        "teacher_forward_count": 0,
        "teacher_cache_scientific_sha256": str(plan["teacher_cache_scientific_sha256"]),
        "descriptor_collection_scientific_sha256": str(
            plan["descriptor_collection_scientific_sha256"]
        ),
        "split_scientific_sha256": str(plan["split_scientific_sha256"]),
        "checkpoint_sha256": str(plan["checkpoint_sha256"]),
        "execution_profile_sha256": str(plan["execution_profile_sha256"]),
        "records": [
            {
                "stable_sample_id": by_id[str(row["stable_sample_id"])].stable_sample_id,
                "split_membership": str(row["split_membership"]),
                "relative_record_path": by_id[
                    str(row["stable_sample_id"])
                ].relative_record_path,
                "contribution_target_record_scientific_sha256": by_id[
                    str(row["stable_sample_id"])
                ].contribution_target_record_scientific_sha256,
                "contribution_target_record_file_sha256": by_id[
                    str(row["stable_sample_id"])
                ].contribution_target_record_file_sha256,
                "verification_status": by_id[
                    str(row["stable_sample_id"])
                ].verification_status,
            }
            for row in ordered_records
        ],
        "gt_map_calibration": {
            "relative_path": calibration_entry.relative_path,
            "scientific_sha256": calibration_entry.scientific_sha256,
            "file_sha256": calibration_entry.file_sha256,
            "verification_status": calibration_entry.verification_status,
        },
        "shapley_normalization": {
            "relative_path": normalization_entry.relative_path,
            "scientific_sha256": normalization_entry.scientific_sha256,
            "file_sha256": normalization_entry.file_sha256,
            "verification_status": normalization_entry.verification_status,
        },
    }
    for identity in SEVEN_LAYERED_IDENTITY_KEYS:
        manifest[identity] = str(plan[identity])
    if repository_identity is not None:
        manifest["repository_identity"] = {
            "contract_tag": repository_identity.contract_tag,
            "contract_commit": repository_identity.contract_commit,
            "generation_commit": repository_identity.generation_commit,
            "head_is_descendant": repository_identity.head_is_descendant,
            "worktree_clean": repository_identity.worktree_clean,
        }
    return manifest


def require_official_materialization_enabled(config: ContributionTargetsConfig) -> None:
    """Fail closed unless the configuration explicitly enables materialization."""

    if not config.official_materialization_enabled:
        _fail(
            "B2_CONTRIBUTION_OFFICIAL_MATERIALIZATION_NOT_ENABLED",
            "official contribution-target materialization is disabled by configuration",
        )
    return None


def _require_expected_plan_sha256(
    config: ContributionTargetsConfig, expected_plan_sha256: Any
) -> str:
    if not config.expected_plan_sha_required_for_official:
        _fail(
            "B2_CONTRIBUTION_CONFIG_DRIFT",
            "official materialization must always require an expected plan hash",
        )
    if expected_plan_sha256 is None:
        _fail(
            "B2_CONTRIBUTION_EXPECTED_PLAN_SHA_MISSING",
            "official materialization requires --expected-plan-sha256",
        )
    if not _is_sha256(expected_plan_sha256):
        _fail(
            "B2_CONTRIBUTION_EXPECTED_PLAN_SHA_MALFORMED",
            "the expected plan hash must be 64 lowercase hex characters",
        )
    return str(expected_plan_sha256)


def materialize_contribution_target_collection(
    *,
    config: ContributionTargetsConfig,
    inputs: Any,
    output_run_dir: Any,
    expected_plan_sha256: Any,
    repository_root: Any | None = None,
    candidate_layers: Sequence[int] | None = None,
    prediction_depths: Sequence[int] | None = None,
) -> ContributionTargetMaterializationResult:
    """Materialize one fresh, fully verified contribution-target run directory."""

    require_official_materialization_enabled(config)
    repository_identity: ContributionRepositoryIdentity | None = None
    repository_path: Path | None = None
    if config.repository_identity_gate_enabled:
        if repository_root is None:
            _fail(
                "B2_CONTRIBUTION_REPOSITORY_ROOT_REQUIRED",
                "official materialization requires a repository root",
            )
        repository_path = Path(repository_root)
        repository_identity = verify_contribution_repository_identity(
            config=config, repository_root=repository_path
        )
    if config.expected_input_artifact_kind == PRODUCTION_ARTIFACT_KIND:
        require_production_artifact_kind(
            {"artifact_kind": getattr(inputs, "artifact_kind", None)}
        )
    elif getattr(inputs, "artifact_kind", None) != config.expected_input_artifact_kind:
        _fail(
            "B2_TARGET_ARTIFACT_KIND_INVALID",
            "input artifact kind does not match the configured expectation",
        )
    expected = _require_expected_plan_sha256(config, expected_plan_sha256)
    run_dir = Path(output_run_dir)
    try:
        refuse_existing_run(run_dir)
    except OutputProtectionError as exc:
        _fail("B2_CONTRIBUTION_OUTPUT_DIR_EXISTS", str(exc))

    collection = run_contribution_target_collection(
        config=config,
        inputs=inputs,
        candidate_layers=candidate_layers,
        prediction_depths=prediction_depths,
    )
    recomputed = str(collection.plan["contribution_plan_scientific_sha256"])
    if recomputed != expected:
        _fail(
            "B2_CONTRIBUTION_EXPECTED_PLAN_SHA_MISMATCH",
            f"recomputed plan hash {recomputed} does not match the expected {expected}",
        )

    if repository_identity is not None:
        assert repository_path is not None
        repository_identity = verify_contribution_repository_identity(
            config=config,
            repository_root=repository_path,
            expected_generation_commit=repository_identity.generation_commit,
        )
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / RECORDS_DIRECTORY).mkdir(parents=True, exist_ok=False)

    record_entries = [
        write_contribution_target_record_atomic(
            run_dir / contribution_record_relative_path(record["stable_sample_id"]),
            record,
            candidate_layers=collection.candidate_layers,
            prediction_depths=collection.prediction_depths,
        )
        for record in collection.records
    ]
    calibration_entry = write_gt_map_calibration_atomic(
        run_dir / CALIBRATION_RELATIVE_PATH, collection.calibration_artifact
    )
    normalization_entry = write_shapley_normalization_atomic(
        run_dir / NORMALIZATION_RELATIVE_PATH, collection.normalization
    )
    manifest = build_contribution_targets_manifest(
        config=config,
        collection=collection,
        record_entries=record_entries,
        calibration_entry=calibration_entry,
        normalization_entry=normalization_entry,
        repository_identity=repository_identity,
    )
    write_final_manifest_with_receipt_atomic(run_dir, manifest)
    verify_contribution_target_collection(config=config, run_dir=run_dir)
    return ContributionTargetMaterializationResult(
        run_dir=run_dir,
        manifest=MappingProxyType(dict(manifest)),
        contribution_plan_scientific_sha256=recomputed,
        teacher_forward_count=0,
    )


def verify_contribution_target_collection(
    *,
    config: ContributionTargetsConfig,
    run_dir: Any,
) -> VerifiedContributionTargetCollection:
    """Verify one materialized contribution-target run directory from disk only."""

    root = Path(run_dir)
    verify_final_manifest_receipt(root)
    manifest_path = root / FINAL_MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail("B2_CONTRIBUTION_MANIFEST_INVALID", f"invalid final manifest: {exc}")
    if not isinstance(manifest, Mapping):
        _fail("B2_CONTRIBUTION_MANIFEST_INVALID", "final manifest must be an object")

    rows = manifest.get("records")
    if not isinstance(rows, Sequence) or not rows:
        _fail("B2_CONTRIBUTION_MANIFEST_INVALID", "final manifest carries no record entries")
    planned_ids = [str(row["stable_sample_id"]) for row in rows]
    audit_contribution_target_integrity(
        run_dir=root, manifest=manifest, planned_ids=planned_ids
    )

    claimed_forwards = manifest.get("teacher_forward_count")
    if isinstance(claimed_forwards, bool) or not isinstance(claimed_forwards, int):
        _fail(
            "B2_CONTRIBUTION_TEACHER_FORWARD_NONZERO",
            "the manifest must record an integer teacher_forward_count",
        )
    teacher_forward_count = int(claimed_forwards)
    if teacher_forward_count != 0:
        _fail(
            "B2_CONTRIBUTION_TEACHER_FORWARD_NONZERO",
            f"the manifest claims {teacher_forward_count} teacher forwards, must be zero",
        )

    layers = tuple(int(layer) for layer in manifest["candidate_layers"])
    depths = tuple(int(depth) for depth in manifest["prediction_depths"])
    records_by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        entry = PersistedTargetRecordEntry(
            stable_sample_id=str(row["stable_sample_id"]),
            relative_record_path=str(row["relative_record_path"]),
            contribution_target_record_scientific_sha256=str(
                row["contribution_target_record_scientific_sha256"]
            ),
            contribution_target_record_file_sha256=str(
                row["contribution_target_record_file_sha256"]
            ),
            verification_status=str(row.get("verification_status", "verified")),
        )
        records_by_id[entry.stable_sample_id] = verify_persisted_contribution_target_record(
            run_dir=root,
            entry=entry,
            candidate_layers=layers,
            prediction_depths=depths,
        )

    calibration_entry = _statistics_entry_from_manifest(
        manifest, "gt_map_calibration", expected_relative_path=CALIBRATION_RELATIVE_PATH
    )
    calibration_path = _resolve_within_root(
        root,
        calibration_entry.relative_path,
        expected_kind="GT map calibration artifact",
        path_code="B2_CONTRIBUTION_RUN_RELATIVE_PATH_INVALID",
        missing_code="B2_CONTRIBUTION_MISSING_ARTIFACT",
        escape_code="B2_CONTRIBUTION_RUN_ROOT_ESCAPE",
    )
    calibration_artifact = _verify_calibration_payload(
        calibration_path, entry=calibration_entry
    )
    normalization_entry = _statistics_entry_from_manifest(
        manifest, "shapley_normalization", expected_relative_path=NORMALIZATION_RELATIVE_PATH
    )
    normalization_path = _resolve_within_root(
        root,
        normalization_entry.relative_path,
        expected_kind="Shapley normalization artifact",
        path_code="B2_CONTRIBUTION_RUN_RELATIVE_PATH_INVALID",
        missing_code="B2_CONTRIBUTION_MISSING_ARTIFACT",
        escape_code="B2_CONTRIBUTION_RUN_ROOT_ESCAPE",
    )
    normalization = _verify_normalization_payload(
        normalization_path, entry=normalization_entry
    )

    replayed = build_contribution_plan(
        records=[records_by_id[stable_id] for stable_id in sorted(records_by_id)],
        calibration_artifact=calibration_artifact,
        normalization=normalization,
        candidate_layers=layers,
        prediction_depths=depths,
        official_materialization_enabled=bool(
            manifest.get("official_materialization_enabled", False)
        ),
    )
    if str(replayed["contribution_plan_scientific_sha256"]) != str(
        manifest["contribution_plan_scientific_sha256"]
    ):
        _fail(
            "B2_CONTRIBUTION_PLAN_HASH_MISMATCH",
            "the replayed plan hash does not match the persisted manifest",
        )
    if bool(manifest.get("official_materialization_enabled")) != bool(
        config.official_materialization_enabled
    ):
        _fail(
            "B2_CONTRIBUTION_MANIFEST_INVALID",
            "manifest official-materialization flag disagrees with the configuration",
        )
    return _seal_verified_collection(
        VerifiedContributionTargetCollection(
            run_dir=root,
            manifest=MappingProxyType(dict(manifest)),
            records_by_id=MappingProxyType(records_by_id),
            calibration_artifact=calibration_artifact,
            normalization=normalization,
            teacher_forward_count=teacher_forward_count,
        )
    )


def _contribution_values_equal(first: Any, second: Any) -> bool:
    if isinstance(first, torch.Tensor) or isinstance(second, torch.Tensor):
        return (
            isinstance(first, torch.Tensor)
            and isinstance(second, torch.Tensor)
            and torch.equal(first, second)
        )
    if isinstance(first, Mapping) or isinstance(second, Mapping):
        if not isinstance(first, Mapping) or not isinstance(second, Mapping):
            return False
        return set(first) == set(second) and all(
            _contribution_values_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, list | tuple) or isinstance(second, list | tuple):
        if not isinstance(first, list | tuple) or not isinstance(second, list | tuple):
            return False
        return len(first) == len(second) and all(
            _contribution_values_equal(left, right)
            for left, right in zip(first, second, strict=True)
        )
    return bool(first == second)


def _contribution_file_hashes(root: Path) -> Mapping[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _contribution_plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _contribution_plain_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_contribution_plain_value(item) for item in value]
    return value


def _contribution_comparison_payload(
    collection: VerifiedContributionTargetCollection,
) -> dict[str, Any]:
    """Freeze one verified collection into a plain comparable payload."""

    return {
        "manifest": _contribution_plain_value(collection.manifest),
        "records_by_id": _contribution_plain_value(collection.records_by_id),
        "calibration_artifact": _contribution_plain_value(collection.calibration_artifact),
        "normalization": _contribution_plain_value(collection.normalization),
        "teacher_forward_count": int(collection.teacher_forward_count),
        "file_hashes": dict(_contribution_file_hashes(collection.run_dir)),
    }


_RECORD_UPSTREAM_IDENTITY_FIELDS = (
    "source_teacher_record_scientific_sha256",
    "descriptor_record_scientific_sha256",
    "teacher_cache_scientific_sha256",
    "teacher_cache_sample_coverage_sha256",
    "descriptor_collection_scientific_sha256",
    "split_scientific_sha256",
    "checkpoint_sha256",
    "execution_profile_sha256",
)
_COALITION_ENUMERATION_FIELDS = ("bitmask", "layer_ids", "coalition_size")
_FAMILY_UTILITY_FIELDS = (
    ("empty_coalition_raw_utility", "empty coalition raw utility"),
    ("grand_coalition_centered_value", "grand coalition centered value"),
    ("utility_mode", "utility mode"),
)


def _compare_coalition_tables(
    *,
    left: Any,
    right: Any,
    sample: str,
    depth: str,
    reasons: list[str],
) -> bool:
    where = f"sample {sample} depth {depth}"
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        reasons.append(f"coalition table shape mismatch for {where}")
        return False
    equal = True
    for left_row, right_row in zip(left, right, strict=True):
        if not isinstance(left_row, Mapping) or not isinstance(right_row, Mapping):
            reasons.append(f"coalition table entry mismatch for {where}")
            equal = False
            continue
        bitmask = left_row.get("bitmask")
        if any(
            not _contribution_values_equal(left_row.get(key), right_row.get(key))
            for key in _COALITION_ENUMERATION_FIELDS
        ):
            reasons.append(f"coalition enumeration mismatch for {where} coalition {bitmask}")
            equal = False
            continue
        for family in TARGET_FAMILIES:
            left_family = left_row.get(family)
            right_family = right_row.get(family)
            if not isinstance(left_family, Mapping) or not isinstance(right_family, Mapping):
                reasons.append(
                    f"coalition family mismatch for {where} family {family} "
                    f"coalition {bitmask}"
                )
                equal = False
                continue
            scope = f"{where} family {family} coalition {bitmask}"
            if not _contribution_values_equal(
                left_family.get("raw_utility"), right_family.get("raw_utility")
            ):
                reasons.append(f"raw utility mismatch for {scope}")
                equal = False
            if not _contribution_values_equal(
                left_family.get("centered_value"), right_family.get("centered_value")
            ):
                reasons.append(f"centered value mismatch for {scope}")
                equal = False
            left_components = left_family.get("utility_components")
            right_components = right_family.get("utility_components")
            if not isinstance(left_components, Mapping) or not isinstance(
                right_components, Mapping
            ):
                reasons.append(f"coalition utility component mismatch for {scope}")
                equal = False
                continue
            for component in sorted(set(left_components) | set(right_components)):
                if not _contribution_values_equal(
                    left_components.get(component), right_components.get(component)
                ):
                    reasons.append(
                        f"coalition utility component mismatch for {scope} "
                        f"component {component}"
                    )
                    equal = False
    return equal


def _compare_contribution_target_payloads(
    *, first: Mapping[str, Any], second: Mapping[str, Any]
) -> ContributionTargetCollectionComparison:
    """Compare two frozen verified payloads field by scientific field."""

    left_manifest = _mapping_or_empty(first.get("manifest"))
    right_manifest = _mapping_or_empty(second.get("manifest"))
    left_records = _mapping_or_empty(first.get("records_by_id"))
    right_records = _mapping_or_empty(second.get("records_by_id"))
    reasons: list[str] = []

    layered_fields = (
        "contribution_plan_scientific_sha256",
        *SEVEN_LAYERED_IDENTITY_KEYS,
    )
    layered_identities_equal = all(
        left_manifest.get(key) == right_manifest.get(key) for key in layered_fields
    )
    if not layered_identities_equal:
        reasons.append("layered scientific identity mismatch")

    left_ids = tuple(sorted(left_records))
    right_ids = tuple(sorted(right_records))
    coverage_equal = True
    if left_ids != right_ids or not _contribution_values_equal(
        left_manifest.get("planned_ordered_stable_sample_ids"),
        right_manifest.get("planned_ordered_stable_sample_ids"),
    ):
        coverage_equal = False
        reasons.append("sample coverage mismatch")
    if not _contribution_values_equal(
        left_manifest.get("split_counts"), right_manifest.get("split_counts")
    ):
        coverage_equal = False
        reasons.append("split count mismatch")

    record_scientific_hashes_equal = True
    utility_tables_equal = True
    signed_shapley_equal = True
    allocations_equal = True
    for stable_id in sorted(set(left_ids) & set(right_ids)):
        left_record = _mapping_or_empty(left_records.get(stable_id))
        right_record = _mapping_or_empty(right_records.get(stable_id))
        if not _contribution_values_equal(
            left_record.get("contribution_target_record_scientific_sha256"),
            right_record.get("contribution_target_record_scientific_sha256"),
        ):
            record_scientific_hashes_equal = False
            reasons.append(f"record scientific hash mismatch for sample {stable_id}")
        for key in _RECORD_UPSTREAM_IDENTITY_FIELDS:
            if not _contribution_values_equal(left_record.get(key), right_record.get(key)):
                record_scientific_hashes_equal = False
                reasons.append(
                    f"upstream identity mismatch for sample {stable_id} field {key}"
                )
        if not _contribution_values_equal(
            left_record.get("split_membership"), right_record.get("split_membership")
        ):
            coverage_equal = False
            reasons.append(f"split membership mismatch for sample {stable_id}")

        left_depths = _mapping_or_empty(left_record.get("depth_targets"))
        right_depths = _mapping_or_empty(right_record.get("depth_targets"))
        for depth in sorted(set(left_depths) | set(right_depths)):
            left_depth = left_depths.get(depth)
            right_depth = right_depths.get(depth)
            if not isinstance(left_depth, Mapping) or not isinstance(right_depth, Mapping):
                utility_tables_equal = False
                signed_shapley_equal = False
                allocations_equal = False
                reasons.append(f"depth target mismatch for sample {stable_id} depth {depth}")
                continue
            where = f"sample {stable_id} depth {depth}"
            if not _contribution_values_equal(
                left_depth.get("ordered_player_layers"),
                right_depth.get("ordered_player_layers"),
            ):
                utility_tables_equal = False
                reasons.append(f"coalition enumeration mismatch for {where}")
            if not _compare_coalition_tables(
                left=left_depth.get("coalition_table"),
                right=right_depth.get("coalition_table"),
                sample=stable_id,
                depth=depth,
                reasons=reasons,
            ):
                utility_tables_equal = False
            for family in TARGET_FAMILIES:
                left_family = _mapping_or_empty(left_depth.get(family))
                right_family = _mapping_or_empty(right_depth.get(family))
                scope = f"{where} family {family}"
                for key, label in _FAMILY_UTILITY_FIELDS:
                    if not _contribution_values_equal(
                        left_family.get(key), right_family.get(key)
                    ):
                        utility_tables_equal = False
                        reasons.append(f"{label} mismatch for {scope}")
                if not _contribution_values_equal(
                    left_family.get("raw_signed_shapley_by_layer"),
                    right_family.get("raw_signed_shapley_by_layer"),
                ):
                    signed_shapley_equal = False
                    reasons.append(f"signed Shapley mismatch for {scope}")
                if not _contribution_values_equal(
                    left_family.get("efficiency_residual"),
                    right_family.get("efficiency_residual"),
                ):
                    signed_shapley_equal = False
                    reasons.append(f"efficiency residual mismatch for {scope}")
                if not _contribution_values_equal(
                    left_family.get("positive_allocation_target_by_layer"),
                    right_family.get("positive_allocation_target_by_layer"),
                ):
                    allocations_equal = False
                    reasons.append(f"allocation mismatch for {scope}")

    gt_calibration_equal = _contribution_values_equal(
        first.get("calibration_artifact"), second.get("calibration_artifact")
    )
    if not gt_calibration_equal:
        reasons.append("GT calibration scientific content mismatch")
    shapley_normalization_equal = _contribution_values_equal(
        first.get("normalization"), second.get("normalization")
    )
    if not shapley_normalization_equal:
        reasons.append("Shapley normalization scientific content mismatch")
    teacher_forward_count_equal = (
        first.get("teacher_forward_count") == 0
        and second.get("teacher_forward_count") == 0
    )
    if not teacher_forward_count_equal:
        reasons.append("teacher forward count must be zero for both collections")
    file_byte_equal = _contribution_values_equal(
        first.get("file_hashes"), second.get("file_hashes")
    )
    scientific_predicates = (
        layered_identities_equal,
        record_scientific_hashes_equal,
        utility_tables_equal,
        signed_shapley_equal,
        allocations_equal,
        gt_calibration_equal,
        shapley_normalization_equal,
        coverage_equal,
        teacher_forward_count_equal,
    )
    return ContributionTargetCollectionComparison(
        scientifically_equivalent=all(scientific_predicates),
        reasons=tuple(dict.fromkeys(reasons)),
        layered_identities_equal=layered_identities_equal,
        record_scientific_hashes_equal=record_scientific_hashes_equal,
        utility_tables_equal=utility_tables_equal,
        signed_shapley_equal=signed_shapley_equal,
        allocations_equal=allocations_equal,
        gt_calibration_equal=gt_calibration_equal,
        shapley_normalization_equal=shapley_normalization_equal,
        coverage_equal=coverage_equal,
        teacher_forward_count_equal=teacher_forward_count_equal,
        file_byte_equal=file_byte_equal,
    )


def _require_verified_collection(value: Any) -> VerifiedContributionTargetCollection:
    if not _is_verified_collection(value):
        _fail(
            "B2_CONTRIBUTION_COLLECTION_NOT_VERIFIED",
            "comparison accepts only collections returned by production verification",
        )
    return value


def compare_contribution_target_collections(
    *,
    first: VerifiedContributionTargetCollection,
    second: VerifiedContributionTargetCollection,
) -> ContributionTargetCollectionComparison:
    """Compare two independently disk-verified collections exactly."""

    return _compare_contribution_target_payloads(
        first=_contribution_comparison_payload(_require_verified_collection(first)),
        second=_contribution_comparison_payload(_require_verified_collection(second)),
    )

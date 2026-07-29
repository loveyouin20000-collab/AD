"""B2-04A dual contribution-target mathematics and artifact contracts.

Story 1 scope: frozen mathematical contracts. Coalition encoding and equal
average fusion, source-training-only GT map calibration, the frozen GT and
teacher utilities, exact Shapley values, and the positive allocation fallback.

Story 2 scope: the per-sample scientific record schema, the GT map calibration
and Shapley normalization artifacts, the layered scientific identities (split
coverage, collection, plan), and the pure leakage-access helpers a future DLCM
loader will use. Every scientific digest comes from an explicit whitelist over
canonical JSON, so paths, timestamps, Git state, runtime attestation, and
file-byte hashes can never enter a scientific identity.

The module is deliberately inert: no artifact persistence, no run directories,
no CLI, no Git or checkpoint inspection, no dataset adapters, and no
target-domain access. Production math is reused rather than reimplemented:
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, NamedTuple, NoReturn

import torch
import torch.nn.functional as functional

import rad.phase_b.b2_teacher_cache as cache_mod
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

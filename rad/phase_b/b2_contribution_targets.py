"""B2-04A dual contribution-target mathematics.

Story 1 scope: frozen mathematical contracts only. Coalition encoding and equal
average fusion, source-training-only GT map calibration, the frozen GT and
teacher utilities, exact Shapley values, and the positive allocation fallback.

The module is deliberately inert: no artifact persistence, no run directories,
no CLI, no Git or checkpoint inspection, no dataset adapters, and no
target-domain access. Production math is reused rather than reimplemented:
Pixel AP comes from ``rad.evaluation.paper_metrics._binary_ap`` and the
full-depth teacher reference comes from ``rad.models.dlcm.sum_preserving_fusion``.

Spatial maps are logically ``[height, width]``. ``as_spatial_map`` also accepts
the teacher-cache shapes ``[1, 1, H, W]`` and ``[1, H, W]`` and always returns a
detached ``float64`` 2-D tensor, so every utility below is evaluated in
``float64`` regardless of the cached storage dtype.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, NamedTuple, NoReturn

import torch
import torch.nn.functional as functional

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
    """Soft Dice on the calibrated map with a linear denominator and ``eps = 1e-6``."""

    prediction = as_spatial_map(calibrated_map, role="calibrated map")
    binary = _require_binary_mask(mask, prediction)
    intersection = float((prediction * binary).sum().item())
    denominator = float(prediction.sum().item()) + float(binary.sum().item()) + SOFT_DICE_EPS
    return float(2.0 * intersection / denominator)


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

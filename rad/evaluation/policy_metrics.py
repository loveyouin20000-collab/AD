from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PolicySampleTrace:
    """Per-sample record exported before aggregate policy metrics."""

    sample_id: str
    selected_depth: int
    residual_gain: float
    pred_mean: float
    pred_suf_prob: float
    target_sufficient: float
    image_score: float
    weights: list[float]
    target_weights: list[float]
    residual_gain_at_decision: float | None = None
    confidence: float | None = None
    risk: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def gain_mae_rmse(
    pred: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if pred.size == 0:
        return float("nan"), float("nan")
    err = pred - target
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return mae, rmse


def beneficial_depth_auroc(
    gains: np.ndarray,
    scores: np.ndarray,
    *,
    epsilon: float,
) -> float:
    """AUROC of predicting beneficial residual gain ``g > epsilon`` from scores."""
    gains = np.asarray(gains, dtype=np.float64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if gains.size == 0:
        return float("nan")
    labels = (gains > float(epsilon)).astype(np.float64)
    if len(np.unique(labels)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if probs.size == 0:
        return float("nan")
    return float(np.mean((probs - labels) ** 2))


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if probs.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (probs >= lo) & (probs < hi)
        else:
            mask = (probs >= lo) & (probs <= hi)
        if not np.any(mask):
            continue
        conf = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        ece += abs(conf - acc) * (float(mask.sum()) / float(probs.size))
    return float(ece)


def false_safe_exit_rate(
    *,
    selected_depths: np.ndarray,
    residual_gains: np.ndarray,
    epsilon: float,
    full_depth: int,
) -> float:
    """P(g_D > epsilon | D < full_depth)."""
    depths = np.asarray(selected_depths, dtype=np.int64).reshape(-1)
    gains = np.asarray(residual_gains, dtype=np.float64).reshape(-1)
    early = depths < int(full_depth)
    if not np.any(early):
        return float("nan")
    return float(np.mean(gains[early] > float(epsilon)))


def false_continue_rate(
    *,
    selected_depths: np.ndarray,
    residual_gains_at_decision: np.ndarray,
    decision_depth: int,
    epsilon: float,
) -> float:
    """Among samples with g_d <= epsilon at decision depth, fraction that continued past it."""
    depths = np.asarray(selected_depths, dtype=np.int64).reshape(-1)
    gains = np.asarray(residual_gains_at_decision, dtype=np.float64).reshape(-1)
    eligible = gains <= float(epsilon)
    if not np.any(eligible):
        return float("nan")
    continued = depths[eligible] > int(decision_depth)
    return float(np.mean(continued))


def risk_coverage_curve(
    *,
    confidence: np.ndarray,
    risk: np.ndarray,
) -> list[dict[str, float]]:
    """Selective-prediction risk-coverage points sorted by descending confidence."""
    conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
    risks = np.asarray(risk, dtype=np.float64).reshape(-1)
    if conf.size == 0:
        return []
    order = np.argsort(-conf)
    risks_sorted = risks[order]
    points: list[dict[str, float]] = []
    for k in range(1, risks_sorted.size + 1):
        points.append(
            {
                "coverage": float(k / risks_sorted.size),
                "risk": float(np.mean(risks_sorted[:k])),
            }
        )
    return points


def contribution_correlations(
    weights: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, float]:
    """Mean Pearson / Spearman correlation across samples (layer vectors)."""
    w = np.asarray(weights, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    if w.ndim != 2 or t.shape != w.shape or w.shape[0] == 0:
        return float("nan"), float("nan")
    pearsons: list[float] = []
    spearmans: list[float] = []
    from scipy.stats import pearsonr, spearmanr

    def _corr_value(result: Any) -> float:
        if hasattr(result, "statistic"):
            return float(result.statistic)
        if hasattr(result, "correlation"):
            return float(result.correlation)
        return float(result[0])

    for i in range(w.shape[0]):
        if np.allclose(w[i], w[i][0]) or np.allclose(t[i], t[i][0]):
            continue
        p = _corr_value(pearsonr(w[i], t[i]))
        s = _corr_value(spearmanr(w[i], t[i]))
        if not np.isnan(p):
            pearsons.append(p)
        if not np.isnan(s):
            spearmans.append(s)
    if not pearsons:
        return float("nan"), float("nan")
    return float(np.mean(pearsons)), float(np.mean(spearmans))


def top_contributing_layer_accuracy(
    weights: np.ndarray,
    targets: np.ndarray,
) -> float:
    w = np.asarray(weights, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    if w.ndim != 2 or t.shape != w.shape or w.shape[0] == 0:
        return float("nan")
    pred = np.argmax(w, axis=1)
    gold = np.argmax(t, axis=1)
    return float(np.mean(pred == gold))


def expected_depth_and_histogram(
    depths: np.ndarray,
) -> tuple[float, dict[int, int]]:
    d = np.asarray(depths, dtype=np.int64).reshape(-1)
    if d.size == 0:
        return float("nan"), {}
    hist = {
        int(k): int(v) for k, v in zip(*np.unique(d, return_counts=True), strict=True)
    }
    return float(np.mean(d)), hist


def aggregate_policy_metrics(
    traces: Sequence[PolicySampleTrace],
    *,
    epsilon: float,
    full_depth: int,
    decision_depth: int,
) -> dict[str, Any]:
    if not traces:
        return {"n": 0}
    pred = np.array([t.pred_mean for t in traces], dtype=np.float64)
    gain = np.array([t.residual_gain for t in traces], dtype=np.float64)
    suf_p = np.array([t.pred_suf_prob for t in traces], dtype=np.float64)
    suf_t = np.array([t.target_sufficient for t in traces], dtype=np.float64)
    depths = np.array([t.selected_depth for t in traces], dtype=np.int64)
    gain_at_dec = np.array(
        [
            t.residual_gain if t.residual_gain_at_decision is None else t.residual_gain_at_decision
            for t in traces
        ],
        dtype=np.float64,
    )
    weights = np.asarray([t.weights for t in traces], dtype=np.float64)
    targets = np.asarray([t.target_weights for t in traces], dtype=np.float64)
    conf = np.array(
        [
            abs(t.image_score - 0.5) if t.confidence is None else float(t.confidence)
            for t in traces
        ],
        dtype=np.float64,
    )
    risk = np.array(
        [
            float(t.residual_gain > epsilon) if t.risk is None else float(t.risk)
            for t in traces
        ],
        dtype=np.float64,
    )

    mae, rmse = gain_mae_rmse(pred, gain)
    pearson, spearman = contribution_correlations(weights, targets)
    expected, hist = expected_depth_and_histogram(depths)
    return {
        "n": len(traces),
        "gain_mae": mae,
        "gain_rmse": rmse,
        "beneficial_depth_auroc": beneficial_depth_auroc(gain, pred, epsilon=epsilon),
        "brier": brier_score(suf_p, suf_t),
        "ece": expected_calibration_error(suf_p, suf_t),
        "false_safe_exit_rate": false_safe_exit_rate(
            selected_depths=depths,
            residual_gains=gain,
            epsilon=epsilon,
            full_depth=full_depth,
        ),
        "false_continue_rate": false_continue_rate(
            selected_depths=depths,
            residual_gains_at_decision=gain_at_dec,
            decision_depth=decision_depth,
            epsilon=epsilon,
        ),
        "risk_coverage": risk_coverage_curve(confidence=conf, risk=risk),
        "contribution_pearson": pearson,
        "contribution_spearman": spearman,
        "top_contributing_layer_accuracy": top_contributing_layer_accuracy(weights, targets),
        "expected_depth": expected,
        "exit_histogram": hist,
    }


def write_policy_traces(
    traces: Sequence[PolicySampleTrace],
    *,
    output_dir: Path | str,
    epsilon: float,
    full_depth: int,
    decision_depth: int,
) -> dict[str, Any]:
    """Write per-sample traces first, then aggregate summary.json."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / "per_sample_traces.jsonl"
    with trace_path.open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(asdict(t)) + "\n")
    summary = aggregate_policy_metrics(
        traces,
        epsilon=epsilon,
        full_depth=full_depth,
        decision_depth=decision_depth,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary

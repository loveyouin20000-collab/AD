from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from rad.evaluation.efficiency import (
    EfficiencyTrace,
    aggregate_efficiency,
    write_efficiency_traces,
)
from rad.evaluation.policy_metrics import (
    PolicySampleTrace,
    aggregate_policy_metrics,
    beneficial_depth_auroc,
    brier_score,
    contribution_correlations,
    expected_calibration_error,
    expected_depth_and_histogram,
    false_continue_rate,
    false_safe_exit_rate,
    gain_mae_rmse,
    risk_coverage_curve,
    top_contributing_layer_accuracy,
    write_policy_traces,
)


def test_gain_mae_rmse_hand_fixture():
    pred = np.array([1.0, 2.0, 3.0])
    tgt = np.array([1.0, 1.0, 5.0])
    mae, rmse = gain_mae_rmse(pred, tgt)
    assert mae == pytest.approx(1.0)
    assert rmse == pytest.approx(math.sqrt(5.0 / 3.0))


def test_beneficial_depth_auroc_hand_fixture():
    # labels: beneficial if gain > 0.05 -> [1, 0, 1, 0]
    gains = np.array([0.10, 0.00, 0.20, 0.01])
    scores = np.array([0.90, 0.10, 0.80, 0.20])
    auroc = beneficial_depth_auroc(gains, scores, epsilon=0.05)
    assert auroc == pytest.approx(1.0)


def test_brier_and_ece_hand_fixture():
    probs = np.array([0.0, 1.0, 1.0, 0.0])
    labels = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier_score(probs, labels) == pytest.approx(0.5)
    ece = expected_calibration_error(
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        n_bins=2,
    )
    assert ece == pytest.approx(0.0)


def test_false_safe_exit_rate_hand_fixture():
    # D=[12,12,24], g=[0.1, 0.0, 0.2], eps=0.05 -> among D<24: rate 1/2
    selected = np.array([12, 12, 24])
    residual_gain = np.array([0.1, 0.0, 0.2])
    rate = false_safe_exit_rate(
        selected_depths=selected,
        residual_gains=residual_gain,
        epsilon=0.05,
        full_depth=24,
    )
    assert rate == pytest.approx(0.5)


def test_false_continue_rate_hand_fixture():
    # decision_depth=12, g=[0.0, 0.0, 0.2], selected=[12, 24, 24]
    # eligible (g<=0.05): first two; false continue: second -> 1/2
    rate = false_continue_rate(
        selected_depths=np.array([12, 24, 24]),
        residual_gains_at_decision=np.array([0.0, 0.0, 0.2]),
        decision_depth=12,
        epsilon=0.05,
    )
    assert rate == pytest.approx(0.5)


def test_risk_coverage_points_hand_fixture():
    conf = np.array([0.9, 0.5, 0.1])
    risk = np.array([0.0, 1.0, 0.0])
    points = risk_coverage_curve(confidence=conf, risk=risk)
    assert len(points) == 3
    assert points[0]["coverage"] == pytest.approx(1 / 3)
    assert points[0]["risk"] == pytest.approx(0.0)
    assert points[1]["coverage"] == pytest.approx(2 / 3)
    assert points[1]["risk"] == pytest.approx(0.5)
    assert points[2]["coverage"] == pytest.approx(1.0)
    assert points[2]["risk"] == pytest.approx(1 / 3)


def test_contribution_correlation_and_top_layer_hand_fixture():
    weights = np.array([[0.5, 0.3, 0.2], [0.1, 0.7, 0.2]])
    targets = np.array([[0.6, 0.3, 0.1], [0.0, 0.9, 0.1]])
    pearson, spearman = contribution_correlations(weights, targets)
    assert pearson > 0.9
    assert spearman > 0.9
    acc = top_contributing_layer_accuracy(weights, targets)
    assert acc == pytest.approx(1.0)


def test_expected_depth_and_histogram_hand_fixture():
    depths = np.array([12, 12, 18, 24])
    expected, hist = expected_depth_and_histogram(depths)
    assert expected == pytest.approx(16.5)
    assert hist == {12: 2, 18: 1, 24: 1}


def test_efficiency_aggregate_hand_fixture():
    traces = [
        EfficiencyTrace(
            sample_id="a",
            latency_ms=10.0,
            selector_overhead_ms=1.0,
            peak_memory_mb=100.0,
        ),
        EfficiencyTrace(
            sample_id="b",
            latency_ms=30.0,
            selector_overhead_ms=3.0,
            peak_memory_mb=200.0,
        ),
    ]
    agg = aggregate_efficiency(traces)
    assert agg["mean_latency_ms"] == pytest.approx(20.0)
    assert agg["throughput_img_s"] == pytest.approx(1000.0 / 20.0)
    assert agg["mean_selector_overhead_ms"] == pytest.approx(2.0)
    assert agg["peak_memory_mb"] == pytest.approx(200.0)


def test_export_traces_before_aggregate_summary(tmp_path: Path):
    traces = [
        PolicySampleTrace(
            sample_id="s0",
            selected_depth=12,
            residual_gain=0.0,
            pred_mean=0.01,
            pred_suf_prob=0.9,
            target_sufficient=1.0,
            image_score=0.1,
            weights=[0.5, 0.5],
            target_weights=[0.6, 0.4],
        ),
        PolicySampleTrace(
            sample_id="s1",
            selected_depth=24,
            residual_gain=0.2,
            pred_mean=0.3,
            pred_suf_prob=0.2,
            target_sufficient=0.0,
            image_score=0.8,
            weights=[0.2, 0.8],
            target_weights=[0.1, 0.9],
        ),
    ]
    out = tmp_path / "eval"
    summary = write_policy_traces(
        traces,
        output_dir=out,
        epsilon=0.05,
        full_depth=24,
        decision_depth=12,
    )
    trace_path = out / "per_sample_traces.jsonl"
    summary_path = out / "summary.json"
    assert trace_path.is_file()
    assert summary_path.is_file()
    assert trace_path.stat().st_mtime <= summary_path.stat().st_mtime
    lines = trace_path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert "false_safe_exit_rate" in summary
    assert "expected_depth" in summary

    eff = [
        EfficiencyTrace("s0", latency_ms=10.0, selector_overhead_ms=1.0, peak_memory_mb=50.0),
        EfficiencyTrace("s1", latency_ms=20.0, selector_overhead_ms=2.0, peak_memory_mb=80.0),
    ]
    eff_summary = write_efficiency_traces(eff, output_dir=out)
    assert (out / "efficiency_traces.jsonl").is_file()
    assert (out / "efficiency_summary.json").is_file()
    assert eff_summary["mean_latency_ms"] == pytest.approx(15.0)


def test_aggregate_policy_metrics_matches_hand_parts():
    traces = [
        PolicySampleTrace(
            sample_id="a",
            selected_depth=12,
            residual_gain=0.0,
            pred_mean=0.0,
            pred_suf_prob=1.0,
            target_sufficient=1.0,
            image_score=0.1,
            weights=[1.0, 0.0],
            target_weights=[1.0, 0.0],
            residual_gain_at_decision=0.0,
        ),
        PolicySampleTrace(
            sample_id="b",
            selected_depth=24,
            residual_gain=0.2,
            pred_mean=0.2,
            pred_suf_prob=0.0,
            target_sufficient=0.0,
            image_score=0.9,
            weights=[0.0, 1.0],
            target_weights=[0.0, 1.0],
            residual_gain_at_decision=0.0,
        ),
    ]
    agg = aggregate_policy_metrics(
        traces, epsilon=0.05, full_depth=24, decision_depth=12
    )
    assert agg["gain_mae"] == pytest.approx(0.0)
    assert agg["false_safe_exit_rate"] == pytest.approx(0.0)
    assert agg["false_continue_rate"] == pytest.approx(0.5)
    assert agg["top_contributing_layer_accuracy"] == pytest.approx(1.0)
    assert agg["n"] == 2

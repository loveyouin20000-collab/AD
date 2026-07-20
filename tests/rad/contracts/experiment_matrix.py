"""Shared experiment-matrix contract assertions."""

from __future__ import annotations

from typing import Any

from rad.evaluation.experiment_matrix import MatrixRow

REQUIRED_SELECTOR_SIGNAL_NAMES: tuple[str, ...] = (
    "response",
    "uncertainty",
    "stability",
    "complementarity",
    "token_separation",
)

FORBIDDEN_EVAL_CLIS: tuple[str, ...] = (
    "tools/smoke_adaptive_engine.py",
    "tools/evaluate_adaptive.py",
)

DATASET_EVAL_CLI = "tools/evaluate_adaptive_dataset.py"

REQUIRED_FIXED_EXIT_ROWS: tuple[str, ...] = (
    "fixed_exit_12_equal",
    "fixed_exit_18_equal",
    "fixed_exit_12_dynamic",
    "fixed_exit_18_dynamic",
)

REQUIRED_SELECTOR_ABLATION_ROWS: tuple[str, ...] = (
    "selector_full",
    "selector_without_response",
    "selector_without_uncertainty",
    "selector_without_stability",
    "selector_without_complementarity",
    "selector_without_token_separation",
)


def _command_text(row: MatrixRow) -> str:
    return " ".join(str(row.command).split())


def assert_dataset_backed_evaluation_row(row: MatrixRow) -> None:
    """Paper evaluation rows must call the dataset-backed evaluator."""
    cmd = _command_text(row)
    assert DATASET_EVAL_CLI in cmd, (
        f"row {row.id}: expected {DATASET_EVAL_CLI} in command, got: {cmd}"
    )
    for forbidden in FORBIDDEN_EVAL_CLIS:
        assert forbidden not in cmd, (
            f"row {row.id}: forbidden CLI {forbidden} appears in command: {cmd}"
        )
    lower = cmd.lower()
    assert "synthetic" not in lower, f"row {row.id}: synthetic evaluation forbidden"
    assert "smoke" not in lower or "smoke_output" not in lower, (
        f"row {row.id}: smoke evaluation CLI forbidden"
    )


def assert_fixed_exit_semantics(row: MatrixRow) -> None:
    """Fixed-exit rows encode exit depth and fusion mode explicitly."""
    method = row.config.get("method") or {}
    assert isinstance(method, dict), f"row {row.id}: method must be a mapping"
    if row.id == "fixed_exit_12_equal":
        assert int(method["exit_depth"]) == 12
        assert method.get("fusion") == "equal"
    elif row.id == "fixed_exit_18_equal":
        assert int(method["exit_depth"]) == 18
        assert method.get("fusion") == "equal"
    elif row.id == "fixed_exit_12_dynamic":
        assert int(method["exit_depth"]) == 12
        assert method.get("fusion") == "dynamic"
    elif row.id == "fixed_exit_18_dynamic":
        assert int(method["exit_depth"]) == 18
        assert method.get("fusion") == "dynamic"
    else:
        raise AssertionError(f"unexpected fixed-exit row id: {row.id}")
    assert_dataset_backed_evaluation_row(row)


def assert_selector_ablation_semantics(row: MatrixRow) -> None:
    """Selector ablation rows must carry a complete explicit signal map."""
    selector = row.config.get("selector") or {}
    assert isinstance(selector, dict), f"row {row.id}: selector must be a mapping"
    signals = selector.get("signals")
    assert isinstance(signals, dict), f"row {row.id}: selector.signals required"
    missing = [n for n in REQUIRED_SELECTOR_SIGNAL_NAMES if n not in signals]
    assert not missing, f"row {row.id}: missing signal keys: {missing}"
    extra = sorted(set(signals) - set(REQUIRED_SELECTOR_SIGNAL_NAMES))
    assert not extra, f"row {row.id}: unknown signal keys: {extra}"
    for name, enabled in signals.items():
        assert isinstance(enabled, bool), (
            f"row {row.id}: selector.signals.{name} must be bool, got {type(enabled)}"
        )

    expected_disabled: str | None = None
    if row.id == "selector_full":
        expected_disabled = None
    elif row.id.startswith("selector_without_"):
        expected_disabled = row.id.removeprefix("selector_without_")
    else:
        raise AssertionError(f"unexpected selector row id: {row.id}")

    for name in REQUIRED_SELECTOR_SIGNAL_NAMES:
        want = False if name == expected_disabled else True
        assert signals[name] is want, (
            f"row {row.id}: expected signals[{name}]={want}, got {signals[name]}"
        )

    assert selector.get("mask_mode", "train_and_infer") == "train_and_infer", (
        f"row {row.id}: primary scientific ablation requires mask_mode=train_and_infer"
    )
    assert_dataset_backed_evaluation_row(row)


def assert_no_superseded_fixed_exit_ids(ids: set[str]) -> None:
    """Old ambiguous fixed-exit IDs must not be reused after the semantics split."""
    superseded = {"fixed_exit_12", "fixed_exit_18"} & ids
    assert not superseded, (
        f"superseded fixed-exit ids must not remain: {sorted(superseded)}"
    )


def assert_no_superseded_selector_ids(ids: set[str]) -> None:
    superseded = {"ablation_selector_cumulative", "ablation_selector_loo"} & ids
    assert not superseded, (
        f"superseded selector ids must not remain: {sorted(superseded)}"
    )


def row_method_config(row: MatrixRow) -> dict[str, Any]:
    method = row.config.get("method") or {}
    if not isinstance(method, dict):
        raise AssertionError(f"row {row.id}: method must be a mapping")
    return method

"""B3-04 conservative early-exit positive-signal contract."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn


class B3ExitTargetRedefinitionError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B3ExitTargetRedefinitionError(code, detail)


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        _fail("B3_EXIT_TARGET_REDEFINITION_JSON_INVALID", f"{path} must contain JSON object")
    return payload


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            _fail("B3_EXIT_TARGET_REDEFINITION_TRACE_INVALID", "JSONL row must be object")
        rows.append(payload)
    if not rows:
        _fail("B3_EXIT_TARGET_REDEFINITION_TRACE_INVALID", "trace must not be empty")
    return rows


def _validate_thresholds(max_gain: float, min_prob: float) -> None:
    if max_gain < 0.0 or max_gain > 0.10:
        _fail(
            "B3_EXIT_TARGET_REDEFINITION_THRESHOLD_UNSAFE",
            "max_predicted_remaining_gain must be in [0, 0.10]",
        )
    if min_prob < 0.50 or min_prob > 1.0:
        _fail(
            "B3_EXIT_TARGET_REDEFINITION_THRESHOLD_UNSAFE",
            "min_predicted_sufficiency_probability must be in [0.50, 1.0]",
        )


def build_positive_signal_contract(
    *,
    calibration_trace_rows: Sequence[Mapping[str, Any]],
    latency_profile: Mapping[str, Any],
    max_predicted_remaining_gain: float,
    min_predicted_sufficiency_probability: float,
    accepted_lse_identity: str,
    b3_02_materialization_identity: str,
    tracked_pt_count: int,
) -> dict[str, Any]:
    if tracked_pt_count != 0:
        _fail("B3_EXIT_TARGET_REDEFINITION_TRACKED_PT", "tracked .pt files must remain zero")
    _validate_thresholds(
        float(max_predicted_remaining_gain),
        float(min_predicted_sufficiency_probability),
    )
    savings = latency_profile.get("depth_savings_proxy_vs_full")
    if not isinstance(savings, Mapping):
        _fail("B3_EXIT_TARGET_REDEFINITION_LATENCY_INVALID", "depth savings proxy required")
    rows: list[dict[str, Any]] = []
    counts = {"12": 0, "18": 0}
    for row in calibration_trace_rows:
        depth = row.get("depth")
        if depth not in (12, 18):
            _fail("B3_EXIT_TARGET_REDEFINITION_TRACE_INVALID", "only early depths are allowed")
        depth_key = str(depth)
        pred_mean = float(row["pred_mean"])
        pred_prob = float(row["pred_suf_prob"])
        saving = float(savings.get(depth_key, 0.0))
        positive = (
            pred_mean <= max_predicted_remaining_gain
            and pred_prob >= min_predicted_sufficiency_probability
            and saving > 0.0
        )
        if positive:
            counts[depth_key] += 1
        rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "depth": int(depth),
                "pred_mean": pred_mean,
                "pred_suf_prob": pred_prob,
                "latency_savings_proxy": saving,
                "positive_signal": positive,
            }
        )
    positive_count = sum(counts.values())
    decision = (
        "positive_signal_contract_ready_pending_materialization"
        if positive_count > 0
        else "no_positive_signal_under_conservative_contract"
    )
    payload = {
        "schema_version": "b3_04_exit_target_positive_signal_contract_v1",
        "decision": decision,
        "training_unlocked": False,
        "training_started": False,
        "evaluation_started": False,
        "final_content_accessed": False,
        "checkpoint_generated": False,
        "accepted_lse_identity": accepted_lse_identity,
        "b3_02_materialization_identity": b3_02_materialization_identity,
        "max_predicted_remaining_gain": float(max_predicted_remaining_gain),
        "min_predicted_sufficiency_probability": float(min_predicted_sufficiency_probability),
        "positive_signal_count": positive_count,
        "candidate_positive_counts_by_depth": counts,
        "records": len(rows),
        "early_depths": [12, 18],
        "full_depth": 24,
        "rows": sorted(rows, key=lambda x: (str(x["sample_id"]), int(x["depth"]))),
        "tracked_pt_count": int(tracked_pt_count),
    }
    payload["positive_signal_contract_identity"] = canonical_json_sha256(payload)
    return payload

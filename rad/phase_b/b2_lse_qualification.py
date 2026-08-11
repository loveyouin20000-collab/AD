"""B2-06E LSE evaluation and qualification helpers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn


class B2LSEQualificationError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2LSEQualificationError(code, detail)


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
        _fail("B2_LSE_QUALIFICATION_JSON_INVALID", f"{path} must contain JSON object")
    return payload


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    return value


def _require_equal(actual: Any, expected: Any, *, key: str) -> None:
    if actual != expected:
        _fail("B2_LSE_QUALIFICATION_IDENTITY_MISMATCH", f"{key} mismatch")


def _finite(value: Any, *, key: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        _fail("B2_LSE_QUALIFICATION_METRIC_INVALID", f"{key} must be finite")
    return out


def _depth_metrics(metrics: Mapping[str, Any], depth: int) -> Mapping[str, Any]:
    item = metrics.get(str(depth))
    if not isinstance(item, Mapping):
        _fail("B2_LSE_QUALIFICATION_METRIC_INVALID", f"missing depth {depth} metrics")
    return item


def qualify_lse_evaluation(
    *,
    receipt: Mapping[str, Any],
    metrics: Mapping[str, Any],
    expected: Mapping[str, Any],
    max_calibration_nll: float,
    required_depths: Sequence[int],
) -> dict[str, Any]:
    if receipt.get("schema_version") != "b2_06d_lse_training_receipt_v1":
        _fail("B2_LSE_QUALIFICATION_RECEIPT_INVALID", "training receipt schema mismatch")
    if receipt.get("training_started") is not True or receipt.get("lse_checkpoint_generated") is not True:
        _fail("B2_LSE_QUALIFICATION_RECEIPT_INVALID", "training receipt must record generated checkpoint")
    for key in (
        "accepted_identity",
        "v5_deployment_identity",
        "H_decision",
        "H_evidence",
        "config_hash",
        "best_checkpoint_sha256",
    ):
        _require_equal(receipt.get(key), expected.get(key), key=key)

    nll = _finite(metrics.get("nll"), key="nll")
    depth_reports: dict[str, dict[str, Any]] = {}
    total_n = 0
    for depth in required_depths:
        depth_metric = _depth_metrics(metrics, int(depth))
        depth_n = int(depth_metric.get("n", 0))
        if depth_n <= 0:
            _fail("B2_LSE_QUALIFICATION_METRIC_INVALID", f"depth {depth} n must be positive")
        total_n += depth_n
        depth_reports[str(int(depth))] = {
            "n": depth_n,
            "nll": _finite(depth_metric.get("nll"), key=f"{depth}.nll"),
            "mae": _finite(depth_metric.get("mae"), key=f"{depth}.mae"),
            "rmse": _finite(depth_metric.get("rmse"), key=f"{depth}.rmse"),
            "brier": _finite(depth_metric.get("brier"), key=f"{depth}.brier"),
            "ece": _finite(depth_metric.get("ece"), key=f"{depth}.ece"),
        }
    threshold = float(max_calibration_nll)
    if nll > threshold:
        _fail(
            "B2_LSE_QUALIFICATION_THRESHOLD_FAILED",
            f"calibration nll {nll} > threshold {threshold}",
        )
    decision = {
        "schema_version": "b2_06e_lse_qualification_decision_v1",
        "verdict": "qualified",
        "accepted_artifact_generated": False,
        "accepted_identity": receipt["accepted_identity"],
        "v5_deployment_identity": receipt["v5_deployment_identity"],
        "unlock_identity": receipt["unlock_identity"],
        "training_receipt_identity": receipt.get("receipt_identity"),
        "best_checkpoint_sha256": receipt["best_checkpoint_sha256"],
        "calibration_nll": nll,
        "max_calibration_nll": threshold,
        "required_depths": [int(depth) for depth in required_depths],
        "evaluated_rows": total_n,
        "depth_metrics": depth_reports,
    }
    decision["H_lse_qualification"] = canonical_json_sha256(decision)
    return decision

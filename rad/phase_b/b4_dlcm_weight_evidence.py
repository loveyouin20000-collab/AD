"""B4-01 DLCM adaptive weight evidence summarization."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, NoReturn

EXPECTED_ACCEPTED_DLCM_IDENTITY = (
    "0c1a411317f212e5deb29040d184d57aead8a6f862fe3146937db99d1f365116"
)
EXPECTED_V5_DEPLOYMENT_IDENTITY = (
    "c56248c9ff6021fc16cf4792d87afeebf1bb8f6d45859f7c26017830dcf0e0bd"
)


class B4DLCMWeightEvidenceError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B4DLCMWeightEvidenceError(code, detail)


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
        _fail("B4_DLCM_WEIGHT_EVIDENCE_JSON_INVALID", f"{path} must contain JSON object")
    return payload


def _require_false(value: Any, detail: str) -> None:
    if value is not False:
        _fail("B4_DLCM_WEIGHT_EVIDENCE_BOUNDARY_VIOLATION", detail)


def _accepted_identities(accepted_reference_evidence: Mapping[str, Any]) -> dict[str, Any]:
    if (
        accepted_reference_evidence.get("schema_version")
        != "b2_06b_accepted_v5_reference_packaging_evidence_v1"
    ):
        _fail("B4_DLCM_WEIGHT_EVIDENCE_SCHEMA_MISMATCH", "accepted reference schema mismatch")
    frozen = accepted_reference_evidence.get("frozen_identities")
    if not isinstance(frozen, Mapping):
        _fail("B4_DLCM_WEIGHT_EVIDENCE_SCHEMA_MISMATCH", "frozen identities missing")
    if frozen.get("accepted_identity") != EXPECTED_ACCEPTED_DLCM_IDENTITY:
        _fail("B4_DLCM_WEIGHT_EVIDENCE_IDENTITY_MISMATCH", "accepted DLCM identity mismatch")
    if frozen.get("v5_deployment_identity") != EXPECTED_V5_DEPLOYMENT_IDENTITY:
        _fail("B4_DLCM_WEIGHT_EVIDENCE_IDENTITY_MISMATCH", "V5 deployment identity mismatch")
    boundary = accepted_reference_evidence.get("boundary", {})
    if isinstance(boundary, Mapping):
        for key in (
            "accepted_identity_changed",
            "accepted_manifest_modified",
            "final_re_evaluated",
            "lse_training_started",
            "lse_checkpoint_generated",
            "push_performed",
        ):
            _require_false(boundary.get(key), f"accepted_reference.{key} must be false")
    return dict(frozen)


def _as_float_list(value: Any, key: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail("B4_DLCM_WEIGHT_EVIDENCE_ROW_INVALID", f"{key} must be a sequence")
    out = [float(v) for v in value]
    if not out:
        _fail("B4_DLCM_WEIGHT_EVIDENCE_ROW_INVALID", f"{key} must not be empty")
    if any((not math.isfinite(v)) or v < -1e-8 for v in out):
        _fail("B4_DLCM_WEIGHT_EVIDENCE_ROW_INVALID", f"{key} must be finite non-negative")
    total = sum(out)
    if abs(total - 1.0) > 1e-5:
        _fail("B4_DLCM_WEIGHT_EVIDENCE_ROW_INVALID", f"{key} must sum to 1")
    return out


def _entropy(weights: Sequence[float]) -> float:
    return -sum(float(w) * math.log(float(w)) for w in weights if float(w) > 0.0)


def _weight_summary(rows: Sequence[Mapping[str, Any]], key: str, tolerance: float) -> dict[str, Any]:
    matrix = [_as_float_list(row[key], key) for row in rows]
    n = len(matrix)
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        _fail("B4_DLCM_WEIGHT_EVIDENCE_ROW_INVALID", f"{key} widths must match")
    uniform = [1.0 / width] * width
    by_layer = [[matrix[i][j] for i in range(n)] for j in range(width)]
    layer_means = [mean(vals) for vals in by_layer]
    layer_stds = [pstdev(vals) if len(vals) > 1 else 0.0 for vals in by_layer]
    sample_linf = [max(abs(v - u) for v, u in zip(row, uniform, strict=True)) for row in matrix]
    sample_l1 = [sum(abs(v - u) for v, u in zip(row, uniform, strict=True)) for row in matrix]
    entropies = [_entropy(row) for row in matrix]
    return {
        "layer_means": layer_means,
        "layer_stds": layer_stds,
        "max_layer_std": max(layer_stds) if layer_stds else 0.0,
        "mean_layer_std": mean(layer_stds) if layer_stds else 0.0,
        "max_sample_linf_delta_from_uniform": max(sample_linf) if sample_linf else 0.0,
        "mean_sample_linf_delta_from_uniform": mean(sample_linf) if sample_linf else 0.0,
        "mean_sample_l1_delta_from_uniform": mean(sample_l1) if sample_l1 else 0.0,
        "mean_entropy": mean(entropies) if entropies else 0.0,
        "uniform_entropy": math.log(width),
        "rows_non_uniform_at_tolerance": sum(1 for v in sample_linf if v > tolerance),
    }


def _sanitize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        dyn = _as_float_list(row.get("dynamic_weights"), "dynamic_weights")
        dep = _as_float_list(row.get("deployment_weights"), "deployment_weights")
        if len(dyn) != len(dep):
            _fail("B4_DLCM_WEIGHT_EVIDENCE_ROW_INVALID", "dynamic/deployment widths mismatch")
        out.append(
            {
                "stable_sample_id": str(row.get("stable_sample_id", "")),
                "category": str(row.get("category", "")),
                "split": str(row.get("split", "calibration")),
                "depth": int(row.get("depth", 24)),
                "player_layer_ids": [int(v) for v in row.get("player_layer_ids", [])],
                "dynamic_weights": dyn,
                "deployment_weights": dep,
            }
        )
    if not out:
        _fail("B4_DLCM_WEIGHT_EVIDENCE_ROW_INVALID", "at least one row required")
    return sorted(out, key=lambda r: (r["split"], r["category"], r["stable_sample_id"]))


def build_weight_evidence_manifest(
    *,
    rows: Sequence[Mapping[str, Any]],
    accepted_reference_evidence: Mapping[str, Any],
    tracked_pt_count: int,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    if tracked_pt_count != 0:
        _fail("B4_DLCM_WEIGHT_EVIDENCE_TRACKED_PT", "tracked .pt files must remain zero")
    frozen = _accepted_identities(accepted_reference_evidence)
    clean_rows = _sanitize_rows(rows)
    dynamic_summary = _weight_summary(clean_rows, "dynamic_weights", tolerance)
    deployment_summary = _weight_summary(clean_rows, "deployment_weights", tolerance)
    uniform_equivalent = deployment_summary["rows_non_uniform_at_tolerance"] == 0
    adaptive_observed = (
        dynamic_summary["rows_non_uniform_at_tolerance"] > 0
        and deployment_summary["rows_non_uniform_at_tolerance"] > 0
        and deployment_summary["max_layer_std"] > tolerance
    )
    payload = {
        "schema_version": "b4_01_dlcm_adaptive_weight_evidence_manifest_v1",
        "phase": "B4-01 DLCM Adaptive Weight Evidence Closure",
        "status": "dlcm_adaptive_weight_evidence_frozen",
        "accepted_dlcm_identity": frozen["accepted_identity"],
        "v5_deployment_identity": frozen["v5_deployment_identity"],
        "accepted_v5_checkpoint_sha256": accepted_reference_evidence.get("checkpoint_sha256"),
        "beta_star_decimal": frozen.get("beta_star_decimal"),
        "split": "calibration",
        "calibration_records": len(clean_rows),
        "candidate_layers": clean_rows[0]["player_layer_ids"],
        "prediction_depth": clean_rows[0]["depth"],
        "uniform_tolerance": float(tolerance),
        "uniform_equivalent_at_tolerance": bool(uniform_equivalent),
        "sample_adaptive_variation_observed": bool(adaptive_observed),
        "dynamic_weight_summary": dynamic_summary,
        "deployment_weight_summary": deployment_summary,
        "category_counts": {
            category: sum(1 for row in clean_rows if row["category"] == category)
            for category in sorted({str(row["category"]) for row in clean_rows})
        },
        "paper_claim": (
            "accepted V5 deployment weights show calibration-split variation across samples"
            if adaptive_observed
            else "accepted V5 deployment weights are not distinguishable from uniform at tolerance"
        ),
        "boundary": {
            "training_started": False,
            "evaluation_started": False,
            "final_content_accessed": False,
            "model_artifact_generated": False,
            "tracked_pt_files": int(tracked_pt_count),
            "pushed": False,
            "pr_opened": False,
        },
    }
    payload["weight_evidence_identity"] = canonical_json_sha256(payload)
    return payload

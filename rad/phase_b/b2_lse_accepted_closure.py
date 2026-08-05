"""B2-06F accepted LSE artifact closure helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn


class B2LSEAcceptedClosureError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2LSEAcceptedClosureError(code, detail)


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
        _fail("B2_LSE_ACCEPTED_CLOSURE_JSON_INVALID", f"{path} must contain JSON object")
    return payload


def _require_equal(actual: Any, expected: Any, *, key: str, code: str) -> None:
    if actual != expected:
        _fail(code, f"{key} mismatch")


def build_accepted_lse_manifest(
    *,
    decision: Mapping[str, Any],
    training_receipt: Mapping[str, Any],
    training_summary: Mapping[str, Any],
    lse_checkpoint_sha256: str,
    accepted_checkpoint_path: str,
    source_checkpoint_path: str,
    closure_git_sha: str,
) -> dict[str, Any]:
    if decision.get("schema_version") != "b2_06e_lse_qualification_decision_v1":
        _fail("B2_LSE_ACCEPTED_CLOSURE_DECISION_INVALID", "decision schema mismatch")
    if decision.get("verdict") != "qualified":
        _fail("B2_LSE_ACCEPTED_CLOSURE_NOT_QUALIFIED", "LSE qualification verdict must be qualified")
    if decision.get("accepted_artifact_generated") is not False:
        _fail(
            "B2_LSE_ACCEPTED_CLOSURE_DECISION_INVALID",
            "06E decision must not already be an accepted artifact",
        )
    if training_receipt.get("schema_version") != "b2_06d_lse_training_receipt_v1":
        _fail("B2_LSE_ACCEPTED_CLOSURE_RECEIPT_INVALID", "training receipt schema mismatch")
    if training_receipt.get("training_started") is not True:
        _fail("B2_LSE_ACCEPTED_CLOSURE_RECEIPT_INVALID", "training receipt must show training_started")
    if training_receipt.get("lse_checkpoint_generated") is not True:
        _fail("B2_LSE_ACCEPTED_CLOSURE_RECEIPT_INVALID", "training receipt must show checkpoint generated")
    for key in (
        "accepted_identity",
        "v5_deployment_identity",
        "unlock_identity",
        "training_receipt_identity",
    ):
        receipt_key = "receipt_identity" if key == "training_receipt_identity" else key
        _require_equal(
            decision.get(key),
            training_receipt.get(receipt_key),
            key=key,
            code="B2_LSE_ACCEPTED_CLOSURE_IDENTITY_MISMATCH",
        )
    _require_equal(
        decision.get("best_checkpoint_sha256"),
        lse_checkpoint_sha256,
        key="best_checkpoint_sha256",
        code="B2_LSE_ACCEPTED_CLOSURE_CHECKPOINT_MISMATCH",
    )
    _require_equal(
        training_receipt.get("best_checkpoint_sha256"),
        lse_checkpoint_sha256,
        key="receipt.best_checkpoint_sha256",
        code="B2_LSE_ACCEPTED_CLOSURE_CHECKPOINT_MISMATCH",
    )
    selector_hash = training_summary.get("selector_signal_layout_hash")
    if not isinstance(selector_hash, str) or not selector_hash:
        _fail(
            "B2_LSE_ACCEPTED_CLOSURE_SELECTOR_IDENTITY_MISSING",
            "training summary missing selector_signal_layout_hash",
        )
    manifest = {
        "schema_version": "b2_06f_lse_accepted_artifact_manifest_v1",
        "lse_qualified": True,
        "accepted_artifact_generated": True,
        "training_started": False,
        "evaluation_started": False,
        "accepted_lse_checkpoint": str(accepted_checkpoint_path),
        "source_lse_checkpoint": str(source_checkpoint_path),
        "lse_checkpoint_sha256": str(lse_checkpoint_sha256),
        "H_lse_qualification": decision["H_lse_qualification"],
        "calibration_nll": decision["calibration_nll"],
        "max_calibration_nll": decision["max_calibration_nll"],
        "evaluated_rows": decision["evaluated_rows"],
        "accepted_dlcm_identity": decision["accepted_identity"],
        "v5_deployment_identity": decision["v5_deployment_identity"],
        "unlock_identity": decision["unlock_identity"],
        "training_receipt_identity": decision["training_receipt_identity"],
        "training_git_sha": training_receipt.get("git_sha"),
        "closure_git_sha": str(closure_git_sha),
        "selector_signal_layout_hash": selector_hash,
        "seed": training_receipt.get("seed"),
        "config_hash": training_receipt.get("config_hash"),
        "upstream": {
            "accepted_dlcm_identity": decision["accepted_identity"],
            "v5_deployment_identity": decision["v5_deployment_identity"],
            "H_lse_qualification": decision["H_lse_qualification"],
            "training_receipt_identity": decision["training_receipt_identity"],
        },
    }
    manifest["accepted_lse_identity"] = canonical_json_sha256(manifest)
    return manifest

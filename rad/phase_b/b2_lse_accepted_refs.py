"""B2-06B accepted V5 reference packaging helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, NoReturn


class B2AcceptedRefsError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2AcceptedRefsError(code, detail)


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _load_json(path: Path, *, code: str) -> dict[str, Any]:
    if not path.is_file():
        _fail(code, f"missing {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        _fail("B2_ACCEPTED_REFS_INVALID_JSON", f"{path} must contain a JSON object")
    return payload


def _require(payload: dict[str, Any], key: str, expected: str, *, code: str) -> None:
    if str(payload.get(key)) != expected:
        _fail(code, f"{key} mismatch")


def package_accepted_checkpoint_reference(
    *,
    accepted_manifest: Path | str,
    source_checkpoint: Path | str,
    source_calibration_manifest: Path | str,
    expected_accepted_identity: str,
    expected_v5_deployment_identity: str,
    expected_calibration_ab_identity: str,
) -> dict[str, Any]:
    accepted_path = Path(accepted_manifest).resolve()
    source_path = Path(source_checkpoint).resolve()
    calibration_path = Path(source_calibration_manifest).resolve()
    if source_path.name != "canonical_deployment_candidate_v5.pt":
        _fail(
            "B2_ACCEPTED_REFS_SOURCE_CHECKPOINT_INVALID",
            "source checkpoint must be canonical_deployment_candidate_v5.pt",
        )
    if not source_path.is_file():
        _fail("B2_ACCEPTED_REFS_SOURCE_CHECKPOINT_MISSING", f"missing {source_path}")

    accepted = _load_json(accepted_path, code="B2_ACCEPTED_REFS_ACCEPTED_MANIFEST_MISSING")
    calibration = _load_json(calibration_path, code="B2_ACCEPTED_REFS_CALIBRATION_MANIFEST_MISSING")
    if accepted.get("schema_version") != "b2_dlcm_v5_accepted_deployment_manifest_v1":
        _fail("B2_ACCEPTED_REFS_ACCEPTED_MANIFEST_INVALID", "accepted manifest schema mismatch")
    if accepted.get("deployment_qualified") is not True:
        _fail("B2_ACCEPTED_REFS_ACCEPTED_MANIFEST_INVALID", "deployment_qualified must be true")

    _require(
        accepted,
        "accepted_identity",
        expected_accepted_identity,
        code="B2_ACCEPTED_REFS_ACCEPTED_IDENTITY_MISMATCH",
    )
    _require(
        accepted,
        "v5_deployment_identity",
        expected_v5_deployment_identity,
        code="B2_ACCEPTED_REFS_DEPLOYMENT_IDENTITY_MISMATCH",
    )
    _require(
        accepted,
        "calibration_ab_identity",
        expected_calibration_ab_identity,
        code="B2_ACCEPTED_REFS_CALIBRATION_IDENTITY_MISMATCH",
    )
    _require(
        calibration,
        "scientific_identity",
        expected_calibration_ab_identity,
        code="B2_ACCEPTED_REFS_CALIBRATION_IDENTITY_MISMATCH",
    )
    selected = calibration.get("selected")
    if not isinstance(selected, dict) or str(selected.get("beta_decimal")) != "0.54":
        _fail("B2_ACCEPTED_REFS_BETA_MISMATCH", "source calibration beta* must be 0.54")

    reference_root = accepted_path.parent / "accepted_refs"
    reference_root.mkdir(parents=True, exist_ok=True)
    destination = reference_root / source_path.name
    shutil.copy2(source_path, destination)
    checkpoint_hash = sha256_file(destination)

    receipt: dict[str, Any] = {
        "schema_version": "b2_06b_accepted_v5_reference_packaging_receipt_v1",
        "accepted_manifest": str(accepted_path),
        "accepted_identity": expected_accepted_identity,
        "accepted_identity_changed": False,
        "v5_deployment_identity": expected_v5_deployment_identity,
        "calibration_ab_identity": expected_calibration_ab_identity,
        "source_checkpoint": str(source_path),
        "packaged_checkpoint": str(destination.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "source_calibration_manifest": str(calibration_path),
        "packaging_mode": "reference_copy_without_manifest_identity_change",
        "training_started": False,
        "lse_checkpoint_generated": False,
    }
    receipt["receipt_identity"] = canonical_json_sha256(dict(receipt))
    receipt_path = reference_root / "b2_06b_accepted_reference_packaging_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (receipt_path.with_suffix(receipt_path.suffix + ".sha256")).write_text(
        sha256_file(receipt_path) + "  " + receipt_path.name + "\n",
        encoding="utf-8",
    )
    return receipt

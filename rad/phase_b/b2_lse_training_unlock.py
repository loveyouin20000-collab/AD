"""B2-06D LSE training unlock contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, NoReturn


class B2LSETrainingUnlockError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2LSETrainingUnlockError(code, detail)


def canonical_json_sha256(payload: dict[str, Any]) -> str:
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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        _fail("B2_LSE_TRAINING_UNLOCK_REQUIRED", f"missing {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        _fail("B2_LSE_TRAINING_UNLOCK_INVALID", "unlock must contain JSON object")
    return payload


def _require_equal(actual: Any, expected: Any, *, key: str, code: str) -> None:
    if actual != expected:
        _fail(code, f"{key} mismatch")


def validate_training_unlock(
    unlock_path: Path | str,
    *,
    preflight: dict[str, Any],
    config_sha256: str,
    train_output_dir: Path | str,
    seed: int,
    epochs: int,
    patience: int,
) -> dict[str, Any]:
    path = Path(unlock_path)
    payload = _load_json(path)
    if payload.get("schema_version") != "b2_06d_lse_training_unlock_v1":
        _fail("B2_LSE_TRAINING_UNLOCK_INVALID", "schema_version mismatch")
    if payload.get("purpose") != "first_controlled_lse_training":
        _fail("B2_LSE_TRAINING_UNLOCK_INVALID", "purpose mismatch")
    if preflight.get("ready") is not True or preflight.get("accepted_gate_passed") is not True:
        _fail("B2_LSE_TRAINING_UNLOCK_PREFLIGHT_NOT_READY", "accepted preflight must be ready")
    for key in ("accepted_identity", "v5_deployment_identity", "H_decision", "H_evidence"):
        _require_equal(
            payload.get(key),
            preflight.get(key),
            key=key,
            code="B2_LSE_TRAINING_UNLOCK_IDENTITY_MISMATCH",
        )
    _require_equal(
        payload.get("config_sha256"),
        config_sha256,
        key="config_sha256",
        code="B2_LSE_TRAINING_UNLOCK_CONFIG_MISMATCH",
    )
    output_dir = Path(train_output_dir).resolve()
    _require_equal(
        str(Path(str(payload.get("train_output_dir", ""))).resolve()),
        str(output_dir),
        key="train_output_dir",
        code="B2_LSE_TRAINING_UNLOCK_OUTPUT_MISMATCH",
    )
    _require_equal(
        int(payload.get("seed", -1)),
        int(seed),
        key="seed",
        code="B2_LSE_TRAINING_UNLOCK_RUN_MISMATCH",
    )
    _require_equal(
        int(payload.get("epochs", -1)),
        int(epochs),
        key="epochs",
        code="B2_LSE_TRAINING_UNLOCK_RUN_MISMATCH",
    )
    _require_equal(
        int(payload.get("patience", -1)),
        int(patience),
        key="patience",
        code="B2_LSE_TRAINING_UNLOCK_RUN_MISMATCH",
    )
    if payload.get("training_started") is not False:
        _fail("B2_LSE_TRAINING_UNLOCK_INVALID", "unlock must declare training_started=false")
    if payload.get("lse_checkpoint_generated") is not False:
        _fail("B2_LSE_TRAINING_UNLOCK_INVALID", "unlock must declare lse_checkpoint_generated=false")
    if (output_dir / "b2_06d_lse_training_receipt.json").exists():
        _fail("B2_LSE_TRAINING_UNLOCK_ALREADY_CONSUMED", "training receipt already exists")
    if (output_dir / "lse_best.pt").exists():
        _fail("B2_LSE_TRAINING_UNLOCK_ALREADY_CONSUMED", "LSE checkpoint already exists")
    identity_payload = dict(payload)
    identity_payload.pop("unlock_identity", None)
    unlock_identity = canonical_json_sha256(identity_payload)
    if payload.get("unlock_identity") not in (None, unlock_identity):
        _fail("B2_LSE_TRAINING_UNLOCK_IDENTITY_MISMATCH", "unlock_identity mismatch")
    return {
        "schema_version": "b2_lse_training_unlock_dry_run_v1",
        "ready": True,
        "training_started": False,
        "unlock_identity": unlock_identity,
        "unlock_path": str(path),
        "train_output_dir": str(output_dir),
        "accepted_identity": payload["accepted_identity"],
        "v5_deployment_identity": payload["v5_deployment_identity"],
        "H_decision": payload["H_decision"],
        "H_evidence": payload["H_evidence"],
    }


def write_training_receipt(
    path: Path | str,
    *,
    unlock_report: dict[str, Any],
    summary: dict[str, Any],
    best_checkpoint_sha256: str,
) -> dict[str, Any]:
    receipt = {
        "schema_version": "b2_06d_lse_training_receipt_v1",
        "unlock_identity": unlock_report["unlock_identity"],
        "accepted_identity": unlock_report["accepted_identity"],
        "v5_deployment_identity": unlock_report["v5_deployment_identity"],
        "H_decision": unlock_report["H_decision"],
        "H_evidence": unlock_report["H_evidence"],
        "training_started": True,
        "lse_checkpoint_generated": True,
        "best_checkpoint": summary["best_checkpoint"],
        "best_checkpoint_sha256": best_checkpoint_sha256,
        "best_cal_nll": summary["best_cal_nll"],
        "epochs_ran": summary["epochs_ran"],
        "config_hash": summary["config_hash"],
        "git_sha": summary["git_sha"],
        "checkpoint_hash": summary["checkpoint_hash"],
        "seed": summary["seed"],
    }
    receipt["receipt_identity"] = canonical_json_sha256(dict(receipt))
    receipt_path = Path(path)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (receipt_path.with_suffix(receipt_path.suffix + ".sha256")).write_text(
        sha256_file(receipt_path) + "  " + receipt_path.name + "\n",
        encoding="utf-8",
    )
    return receipt

"""B2-05C2 V3 protocol: error codes, gates, final-access forbid."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn

ERROR_CODES: tuple[str, ...] = (
    "B2_DLCM_V3_REAL_TRAINING_NOT_ENABLED",
    "B2_DLCM_V3_CONTRACT_MISMATCH",
    "B2_DLCM_CATEGORY_BATCH_INVALID",
    "B2_DLCM_CATEGORY_COVERAGE_INVALID",
    "B2_DLCM_SMOOTHMAX_INVALID",
    "B2_DLCM_NO_ELIGIBLE_CHECKPOINT",
    "B2_DLCM_ROSTER_ADOPTION_MISMATCH",
    "B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN",
    "B2_DLCM_DEVELOPMENT_UNQUALIFIED",
    "B2_DLCM_FINAL_MATERIALIZATION_MISMATCH",
    "B2_DLCM_FINAL_EVALUATION_MISMATCH",
    "B2_DLCM_ACCEPTED_MANIFEST_FORBIDDEN",
)

SCHEMA_VERSION = "b2_dlcm_v3_protocol_v1"
FORBIDDEN_BYPASS_FLAGS = (
    "force_unlock",
    "bypass_gates",
    "skip_development",
    "allow_final_without_development",
    "ignore_auxiliary_diagnostics",
)


class B2DLCMV3ProtocolError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMV3ProtocolError(code, detail)


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def reject_bypass_flags(config: Mapping[str, Any]) -> None:
    for flag in FORBIDDEN_BYPASS_FLAGS:
        if flag in config and config[flag]:
            _fail("B2_DLCM_V3_CONTRACT_MISMATCH", f"bypass flag forbidden: {flag}")


def require_real_training_enabled(config: Mapping[str, Any], *, dry_run: bool) -> None:
    reject_bypass_flags(config)
    if dry_run:
        return
    if config.get("real_training_enabled") is not True:
        _fail(
            "B2_DLCM_V3_REAL_TRAINING_NOT_ENABLED",
            "real training disabled by contract",
        )


def forbid_final_content_access(*, unlocked: bool, context: str) -> None:
    if not unlocked:
        _fail(
            "B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN",
            f"final content access forbidden before unlock ({context})",
        )


def forbid_accepted_manifest(*, allowed: bool, context: str) -> None:
    if not allowed:
        _fail(
            "B2_DLCM_ACCEPTED_MANIFEST_FORBIDDEN",
            f"accepted manifest forbidden ({context})",
        )


def verify_json_receipt(path: Path | str) -> dict[str, Any]:
    artifact = Path(path)
    if not artifact.is_file():
        _fail("B2_DLCM_V3_CONTRACT_MISMATCH", f"missing artifact {artifact}")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    receipt = artifact.with_suffix(artifact.suffix + ".sha256")
    if not receipt.is_file():
        _fail("B2_DLCM_V3_CONTRACT_MISMATCH", f"missing receipt {receipt}")
    claimed = receipt.read_text(encoding="utf-8").strip().split()[0]
    actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if claimed != actual:
        _fail("B2_DLCM_V3_CONTRACT_MISMATCH", f"receipt mismatch for {artifact}")
    return payload


def write_json_with_receipt(path: Path | str, payload: Mapping[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    target.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(target.suffix + ".sha256").write_text(f"{digest}  {target.name}\n", encoding="utf-8")
    return digest


def persist_json_atomic(path: Path | str, payload: Mapping[str, Any]) -> str:
    """Atomic JSON + receipt write used by authoritative runners."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest

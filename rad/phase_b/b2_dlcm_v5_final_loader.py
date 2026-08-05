"""Formal accepted-loader guard for V5 Final artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NoReturn


class B2DLCMV5FinalLoaderError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMV5FinalLoaderError(code, detail)


def verify_accepted_v5_final_manifest(
    manifest: Mapping[str, Any] | None,
    *,
    expected: Mapping[str, Any],
) -> None:
    if not isinstance(manifest, Mapping):
        _fail("B2_DLCM_NOT_ACCEPTED", "accepted V5 final manifest required")
    if manifest.get("schema_version") != "b2_dlcm_v5_accepted_deployment_manifest_v1":
        _fail("B2_DLCM_NOT_ACCEPTED", "accepted manifest schema mismatch")
    if manifest.get("deployment_qualified") is not True:
        _fail("B2_DLCM_NOT_ACCEPTED", "deployment_qualified must be true")
    for key in (
        "v5_deployment_identity",
        "beta_star_decimal",
        "calibration_ab_identity",
        "H_decision",
        "H_evidence",
        "accepted_identity",
    ):
        if not manifest.get(key):
            _fail("B2_DLCM_NOT_ACCEPTED", f"missing {key}")
    if manifest.get("v5_deployment_identity") != expected.get("v5_deployment_identity"):
        _fail("B2_DLCM_NOT_ACCEPTED", "V5 deployment identity mismatch")
    if manifest.get("beta_star_decimal") != "0.54":
        _fail("B2_DLCM_NOT_ACCEPTED", "beta* mismatch")
    if manifest.get("calibration_ab_identity") != expected.get("calibration_ab_identity"):
        _fail("B2_DLCM_NOT_ACCEPTED", "calibration identity mismatch")

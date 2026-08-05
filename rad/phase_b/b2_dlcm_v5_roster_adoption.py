"""Adopt untouched C1 final roster for V5 without reselection."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn

from rad.phase_b import b2_dlcm_v5 as v5
from rad.phase_b import b2_dlcm_v5_protocol as protocol

C1_ROSTER_RELATIVE = Path("docs/phase_b/b2_05c1_final_evaluation_roster.json")
EXPECTED_ROSTER_SCIENTIFIC = v5.ADOPTED_ROSTER_SCIENTIFIC
ADOPTION_SCHEMA = "b2_dlcm_v5_final_roster_adoption_manifest_v1"


class B2DLCMV5RosterAdoptionError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMV5RosterAdoptionError(code, detail)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_and_verify_c1_roster(repo_root: Path | str) -> dict[str, Any]:
    root = Path(repo_root)
    roster_path = root / C1_ROSTER_RELATIVE
    if not roster_path.is_file():
        _fail("B2_DLCM_V5_ROSTER_ADOPTION_MISMATCH", f"missing roster {roster_path}")
    receipt_path = Path(str(roster_path) + ".sha256")
    if not receipt_path.is_file():
        _fail("B2_DLCM_V5_ROSTER_ADOPTION_MISMATCH", "missing roster receipt")
    claimed = receipt_path.read_text(encoding="utf-8").strip().split()[0]
    actual = _sha256_file(roster_path)
    if claimed != actual:
        _fail("B2_DLCM_V5_ROSTER_ADOPTION_MISMATCH", "roster receipt mismatch")
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    if roster.get("roster_scientific_sha256") != EXPECTED_ROSTER_SCIENTIFIC:
        _fail("B2_DLCM_V5_ROSTER_ADOPTION_MISMATCH", "roster scientific identity mismatch")
    if roster.get("final_content_resolved") is not False:
        _fail("B2_DLCM_V5_ROSTER_ADOPTION_MISMATCH", "final_content_resolved must be false")
    if roster.get("paths_present") is not False:
        _fail("B2_DLCM_V5_ROSTER_ADOPTION_MISMATCH", "paths_present must be false")
    records = roster.get("records")
    if not isinstance(records, list) or len(records) != 16:
        _fail("B2_DLCM_V5_ROSTER_ADOPTION_MISMATCH", "expected 16 records")
    _assert_no_path_fields(roster)
    return roster


def _assert_no_path_fields(payload: Any, *, path: str = "$") -> None:
    allowed_keys = {
        "paths_present",
        "source_roster_relative_path",
    }
    forbidden_exact = {
        "path",
        "paths",
        "uri",
        "url",
        "filename",
        "directory",
        "dir",
        "image_path",
        "mask_path",
    }
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_l = str(key).lower()
            if key_l in forbidden_exact or (
                key_l not in allowed_keys
                and any(f == key_l or key_l.endswith("_" + f) for f in ("path", "uri", "url", "filename"))
            ):
                _fail("B2_DLCM_V5_ROSTER_ADOPTION_MISMATCH", f"forbidden key {path}.{key}")
            if isinstance(value, str) and ("/" in value or "\\" in value):
                if any(tok in value for tok in (".png", ".jpg", "/root/", "MVTec", "http")):
                    _fail("B2_DLCM_V5_ROSTER_ADOPTION_MISMATCH", f"path-like value at {path}.{key}")
            _assert_no_path_fields(value, path=f"{path}.{key}")
    elif isinstance(payload, list):
        for i, item in enumerate(payload):
            _assert_no_path_fields(item, path=f"{path}[{i}]")


def ordered_stable_ids(roster: Mapping[str, Any]) -> list[str]:
    return [str(row["stable_sample_id"]) for row in roster["records"]]


def prove_no_final_access_artifacts(repo_root: Path | str) -> dict[str, bool]:
    root = Path(repo_root)
    patterns = (
        "*final*materialization*unlock*",
        "*final*evaluation*unlock*",
        "*final*materialization*receipt*",
        "*accepted*manifest*",
    )
    found: list[str] = []
    search_roots = [root / "docs" / "phase_b", root / "artifacts" / "phase_b"]
    for base in search_roots:
        if not base.exists():
            continue
        for pat in patterns:
            for hit in base.rglob(pat):
                if hit.suffix in {".md"}:
                    continue
                if "contribution_target_materialization" in str(hit):
                    continue
                found.append(str(hit))
    if found:
        _fail(
            "B2_DLCM_V5_ROSTER_ADOPTION_MISMATCH",
            f"unexpected final access artifacts: {found[:5]}",
        )
    return {
        "final_materialization_unlock_present": False,
        "final_materialization_receipt_present": False,
        "final_evaluation_unlock_present": False,
        "accepted_manifest_present": False,
    }


def build_adoption_manifest(
    *,
    repo_root: Path | str,
    implementation_commit: str,
    v5_contract_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(repo_root)
    roster = load_and_verify_c1_roster(root)
    receipt_path = root / C1_ROSTER_RELATIVE
    roster_file_sha = _sha256_file(receipt_path)
    proofs = prove_no_final_access_artifacts(root)
    final_forbid_ok = False
    try:
        protocol.forbid_final_content_access(unlocked=False, context="adoption_proof")
    except protocol.B2DLCMV5ProtocolError as exc:
        final_forbid_ok = exc.code == "B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN"
    if not final_forbid_ok:
        _fail("B2_DLCM_V5_ROSTER_ADOPTION_MISMATCH", "final access forbid inactive")

    identity = dict(v5_contract_identity or {})
    identity.setdefault("architecture_contract_version", v5.ARCHITECTURE_CONTRACT_VERSION)
    identity.setdefault("model_class_id", v5.MODEL_CLASS_ID)
    identity.setdefault("schema_version", v5.SCHEMA_VERSION)
    identity.setdefault("calibration_contract_version", v5.CALIBRATION_CONTRACT_VERSION)

    manifest = {
        "schema_version": ADOPTION_SCHEMA,
        "selection_reused_without_change": True,
        "source_roster_relative_path": str(C1_ROSTER_RELATIVE).replace("\\", "/"),
        "source_roster_scientific_sha256": EXPECTED_ROSTER_SCIENTIFIC,
        "source_roster_file_sha256": roster_file_sha,
        "source_roster_receipt_sha256": hashlib.sha256(
            (root / (str(C1_ROSTER_RELATIVE) + ".sha256")).read_bytes()
        ).hexdigest(),
        "ordered_stable_sample_ids": ordered_stable_ids(roster),
        "counts": roster.get("counts"),
        "final_content_resolved": False,
        "paths_present": False,
        "implementation_commit": implementation_commit,
        "v5_contract_identity": identity,
        "v1_immutable": v5.v1_immutable_identity(),
        "v2_immutable": v5.v2_immutable_identity(),
        "v3_immutable": v5.v3_immutable_identity(),
        "v4_immutable": v5.v4_immutable_identity(),
        "proofs": {
            **proofs,
            "final_content_access_forbidden": True,
            "roster_records_not_rewritten": True,
        },
    }
    raw_ids = manifest["ordered_stable_sample_ids"]
    if not isinstance(raw_ids, list):
        _fail("B2_DLCM_V5_ROSTER_ADOPTION_MISMATCH", "ordered ids must be a list")
    ordered_ids = [str(x) for x in raw_ids]
    if len(ordered_ids) != 16:
        _fail("B2_DLCM_V5_ROSTER_ADOPTION_MISMATCH", "ordered ids length mismatch")
    if len(set(ordered_ids)) != 16:
        _fail("B2_DLCM_V5_ROSTER_ADOPTION_MISMATCH", "ordered ids not unique")
    return manifest


def assert_adoption_matches_roster(
    manifest: Mapping[str, Any],
    roster: Mapping[str, Any],
) -> None:
    if manifest.get("selection_reused_without_change") is not True:
        _fail("B2_DLCM_V5_ROSTER_ADOPTION_MISMATCH", "selection_reused_without_change must be true")
    if manifest.get("source_roster_scientific_sha256") != roster.get("roster_scientific_sha256"):
        _fail("B2_DLCM_V5_ROSTER_ADOPTION_MISMATCH", "scientific identity drift")
    if ordered_stable_ids(roster) != list(manifest.get("ordered_stable_sample_ids", [])):
        _fail("B2_DLCM_V5_ROSTER_ADOPTION_MISMATCH", "ordered stable ids mutated")

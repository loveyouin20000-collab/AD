"""B3-03 early-exit training contract and no-positive handling."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn


class B3ExitTrainingContractError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B3ExitTrainingContractError(code, detail)


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
        _fail("B3_EXIT_TRAINING_CONTRACT_JSON_INVALID", f"{path} must contain JSON object")
    return payload


def _require_false(value: Any, key: str) -> None:
    if value is not False:
        _fail("B3_EXIT_TRAINING_CONTRACT_BOUNDARY_VIOLATION", f"{key} must be false")


def build_exit_training_contract(
    *,
    prerequisite_manifest: Mapping[str, Any],
    git_sha: str,
    tracked_pt_count: int,
) -> dict[str, Any]:
    if prerequisite_manifest.get("schema_version") != "b3_02_exit_prerequisite_materialization_manifest_v1":
        _fail("B3_EXIT_TRAINING_CONTRACT_PREREQ_INVALID", "B3-02 manifest schema mismatch")
    if tracked_pt_count != 0:
        _fail("B3_EXIT_TRAINING_CONTRACT_TRACKED_PT", "tracked .pt files must remain zero")
    _require_false(prerequisite_manifest.get("training_started"), "B3-02.training_started")
    _require_false(prerequisite_manifest.get("evaluation_started"), "B3-02.evaluation_started")
    _require_false(prerequisite_manifest.get("final_content_accessed"), "B3-02.final_content_accessed")
    _require_false(prerequisite_manifest.get("checkpoint_generated"), "B3-02.checkpoint_generated")
    exits = prerequisite_manifest.get("target_exits_by_depth")
    if not isinstance(exits, Mapping):
        _fail("B3_EXIT_TRAINING_CONTRACT_PREREQ_INVALID", "target_exits_by_depth required")
    positive_exits = sum(int(v) for v in exits.values())
    if positive_exits == 0:
        decision = "conservative_full_depth_fallback"
        reason = "no_positive_exit_targets"
    else:
        decision = "training_contract_ready_pending_unlock"
        reason = "positive_exit_targets_present"
    payload = {
        "schema_version": "b3_03_exit_policy_training_contract_v1",
        "decision": decision,
        "reason": reason,
        "training_unlocked": False,
        "training_started": False,
        "evaluation_started": False,
        "final_content_accessed": False,
        "checkpoint_generated": False,
        "fallback_depth": 24,
        "early_depths": [12, 18],
        "positive_exit_targets": positive_exits,
        "target_exits_by_depth": dict(exits),
        "records": prerequisite_manifest.get("records"),
        "accepted_dlcm_identity": prerequisite_manifest.get("accepted_dlcm_identity"),
        "accepted_lse_identity": prerequisite_manifest.get("accepted_lse_identity"),
        "b2_phase_final_closure_identity": prerequisite_manifest.get("b2_phase_final_closure_identity"),
        "b3_02_materialization_identity": prerequisite_manifest.get("materialization_identity"),
        "git_sha": str(git_sha),
        "tracked_pt_count": int(tracked_pt_count),
    }
    payload["training_contract_identity"] = canonical_json_sha256(payload)
    return payload

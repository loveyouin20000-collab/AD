"""B3-05 early-exit line closure and negative result evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn


class B3EarlyExitLineClosureError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B3EarlyExitLineClosureError(code, detail)


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
        _fail("B3_EARLY_EXIT_LINE_CLOSURE_JSON_INVALID", f"{path} must contain JSON object")
    return payload


def _require_schema(payload: Mapping[str, Any], expected: str, label: str) -> None:
    if payload.get("schema_version") != expected:
        _fail("B3_EARLY_EXIT_LINE_CLOSURE_SCHEMA_MISMATCH", f"{label} schema mismatch")


def _require_false(payload: Mapping[str, Any], key: str, label: str) -> None:
    if payload.get(key) is not False:
        _fail("B3_EARLY_EXIT_LINE_CLOSURE_BOUNDARY_VIOLATION", f"{label}.{key} must be false")


def _require_equal(left: Any, right: Any, detail: str) -> None:
    if left != right:
        _fail("B3_EARLY_EXIT_LINE_CLOSURE_IDENTITY_MISMATCH", detail)


def _sum_counts(counts: Any, key: str) -> int:
    if not isinstance(counts, Mapping):
        _fail("B3_EARLY_EXIT_LINE_CLOSURE_COUNT_INVALID", f"{key} must be a mapping")
    return sum(int(v) for v in counts.values())


def _check_common_boundaries(payload: Mapping[str, Any], label: str) -> None:
    for key in (
        "training_unlocked",
        "training_started",
        "evaluation_started",
        "final_content_accessed",
        "checkpoint_generated",
    ):
        if key in payload:
            _require_false(payload, key, label)


def build_early_exit_line_closure(
    *,
    b3_02_manifest: Mapping[str, Any],
    b3_03_training_contract: Mapping[str, Any],
    b3_04_positive_signal_contract: Mapping[str, Any],
    tracked_pt_count: int,
) -> dict[str, Any]:
    if tracked_pt_count != 0:
        _fail("B3_EARLY_EXIT_LINE_CLOSURE_TRACKED_PT", "tracked .pt files must remain zero")
    _require_schema(
        b3_02_manifest,
        "b3_02_exit_prerequisite_materialization_manifest_v1",
        "B3-02 manifest",
    )
    _require_schema(
        b3_03_training_contract,
        "b3_03_exit_policy_training_contract_v1",
        "B3-03 training contract",
    )
    _require_schema(
        b3_04_positive_signal_contract,
        "b3_04_exit_target_positive_signal_contract_v1",
        "B3-04 positive-signal contract",
    )
    _check_common_boundaries(b3_02_manifest, "B3-02")
    _check_common_boundaries(b3_03_training_contract, "B3-03")
    _check_common_boundaries(b3_04_positive_signal_contract, "B3-04")

    b3_02_identity = b3_02_manifest.get("materialization_identity")
    _require_equal(
        b3_03_training_contract.get("b3_02_materialization_identity"),
        b3_02_identity,
        "B3-03 must bind B3-02 materialization identity",
    )
    _require_equal(
        b3_04_positive_signal_contract.get("b3_02_materialization_identity"),
        b3_02_identity,
        "B3-04 must bind B3-02 materialization identity",
    )
    _require_equal(
        b3_03_training_contract.get("accepted_lse_identity"),
        b3_04_positive_signal_contract.get("accepted_lse_identity"),
        "B3-03 and B3-04 accepted LSE identities must match",
    )
    _require_equal(
        b3_02_manifest.get("accepted_lse_identity"),
        b3_04_positive_signal_contract.get("accepted_lse_identity"),
        "B3-02 and B3-04 accepted LSE identities must match",
    )
    _require_equal(
        b3_02_manifest.get("accepted_dlcm_identity"),
        b3_03_training_contract.get("accepted_dlcm_identity"),
        "B3-02 and B3-03 accepted DLCM identities must match",
    )
    _require_equal(
        b3_02_manifest.get("b2_phase_final_closure_identity"),
        b3_03_training_contract.get("b2_phase_final_closure_identity"),
        "B3-02 and B3-03 B2 closure identities must match",
    )

    target_exit_count = _sum_counts(b3_02_manifest.get("target_exits_by_depth"), "B3-02 target exits")
    contract_target_count = int(b3_03_training_contract.get("positive_exit_targets", -1))
    positive_signal_count = int(b3_04_positive_signal_contract.get("positive_signal_count", -1))
    positive_signal_depth_count = _sum_counts(
        b3_04_positive_signal_contract.get("candidate_positive_counts_by_depth"),
        "B3-04 candidate positives",
    )
    if target_exit_count != 0 or contract_target_count != 0:
        _fail("B3_EARLY_EXIT_LINE_CLOSURE_EXIT_TARGET_PRESENT", "exit targets are present")
    if positive_signal_count != 0 or positive_signal_depth_count != 0:
        _fail(
            "B3_EARLY_EXIT_LINE_CLOSURE_POSITIVE_SIGNAL_PRESENT",
            "positive early-exit signal is present",
        )
    if b3_03_training_contract.get("decision") != "conservative_full_depth_fallback":
        _fail("B3_EARLY_EXIT_LINE_CLOSURE_DECISION_INVALID", "B3-03 decision must be fallback")
    if (
        b3_04_positive_signal_contract.get("decision")
        != "no_positive_signal_under_conservative_contract"
    ):
        _fail(
            "B3_EARLY_EXIT_LINE_CLOSURE_DECISION_INVALID",
            "B3-04 decision must be no-positive-signal",
        )

    payload = {
        "schema_version": "b3_05_early_exit_line_closure_manifest_v1",
        "status": "early_exit_line_closed_negative_result",
        "decision": "conservative_full_depth_fallback",
        "reason": "no_legal_positive_exit_signal",
        "accepted_dlcm_identity": b3_02_manifest.get("accepted_dlcm_identity"),
        "accepted_lse_identity": b3_02_manifest.get("accepted_lse_identity"),
        "b2_phase_final_closure_identity": b3_02_manifest.get("b2_phase_final_closure_identity"),
        "b3_02_materialization_identity": b3_02_identity,
        "b3_03_training_contract_identity": b3_03_training_contract.get(
            "training_contract_identity"
        ),
        "b3_04_positive_signal_contract_identity": b3_04_positive_signal_contract.get(
            "positive_signal_contract_identity"
        ),
        "records": int(b3_02_manifest.get("records", 0)),
        "early_depths": list(b3_04_positive_signal_contract.get("early_depths", [12, 18])),
        "fallback_depth": int(b3_03_training_contract.get("fallback_depth", 24)),
        "target_exits_by_depth": dict(b3_02_manifest.get("target_exits_by_depth", {})),
        "candidate_positive_counts_by_depth": dict(
            b3_04_positive_signal_contract.get("candidate_positive_counts_by_depth", {})
        ),
        "positive_exit_targets": contract_target_count,
        "positive_signal_count": positive_signal_count,
        "training_unlocked": False,
        "training_started": False,
        "evaluation_started": False,
        "final_content_accessed": False,
        "checkpoint_generated": False,
        "tracked_pt_count": int(tracked_pt_count),
    }
    payload["line_closure_identity"] = canonical_json_sha256(payload)
    return payload

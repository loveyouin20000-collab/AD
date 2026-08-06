"""B3-02 early-exit prerequisite materialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, NoReturn

from rad.phase_b import b3_early_exit_gate as gate


class B3ExitPrerequisiteError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B3ExitPrerequisiteError(code, detail)


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    blob = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _as_path(value: Any, *, repo_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        _fail("B3_EXIT_PREREQ_CONFIG_INVALID", "path value required")
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        _fail("B3_EXIT_PREREQ_CALIBRATION_PREDICTIONS_REQUIRED", f"missing {path}")
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            _fail("B3_EXIT_PREREQ_CALIBRATION_PREDICTIONS_INVALID", f"line {lineno} invalid")
        rows.append(payload)
    if not rows:
        _fail("B3_EXIT_PREREQ_CALIBRATION_PREDICTIONS_INVALID", "no prediction rows")
    return rows


def _normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        sample_id = row.get("sample_id")
        depth = row.get("depth")
        if not isinstance(sample_id, str) or not isinstance(depth, int):
            _fail("B3_EXIT_PREREQ_CALIBRATION_PREDICTIONS_INVALID", "sample_id/depth required")
        if depth not in (12, 18):
            _fail("B3_EXIT_PREREQ_CALIBRATION_PREDICTIONS_INVALID", "only early depths allowed")
        target_sufficient = float(row.get("target_sufficient", 0.0))
        normalized.append(
            {
                "sample_id": sample_id,
                "depth": depth,
                "target_gain": float(row["target_gain"]),
                "target_sufficient": target_sufficient,
                "target_exit": bool(target_sufficient >= 0.5),
                "pred_mean": float(row["pred_mean"]),
                "pred_suf_prob": float(row["pred_suf_prob"]),
            }
        )
    return sorted(normalized, key=lambda x: (str(x["sample_id"]), int(x["depth"])))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def materialize_exit_prerequisites(
    *,
    config_path: Path | str,
    calibration_predictions_path: Path | str,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    cfg = gate.load_early_exit_preflight_config(config_path, repo_root=root)
    chain = gate.validate_accepted_b2_chain(cfg)
    rows = _normalize_rows(_read_predictions(Path(calibration_predictions_path)))
    exit_target_path = _as_path(cfg.values["exit_target_manifest"], repo_root=root)
    latency_path = _as_path(cfg.values["latency_profile"], repo_root=root)
    trace_path = _as_path(cfg.values["calibration_trace"], repo_root=root)

    counts_by_depth = {
        str(depth): sum(1 for row in rows if row["depth"] == depth)
        for depth in (12, 18)
    }
    exits_by_depth = {
        str(depth): sum(1 for row in rows if row["depth"] == depth and row["target_exit"])
        for depth in (12, 18)
    }
    target_payload = {
        "schema_version": "b3_02_exit_target_manifest_v1",
        "source": "b2_06d_lse_calibration_predictions",
        "accepted_lse_identity": chain["accepted_lse_manifest"]["accepted_lse_identity"],
        "b2_phase_final_closure_identity": chain["b2_closure_manifest"]["phase_final_closure_identity"],
        "early_depths": [12, 18],
        "full_depth": 24,
        "records": rows,
        "counts_by_depth": counts_by_depth,
        "target_exits_by_depth": exits_by_depth,
        "training_started": False,
        "evaluation_started": False,
    }
    target_payload["target_manifest_identity"] = canonical_json_sha256(target_payload)

    latency_payload = {
        "schema_version": "b3_02_latency_profile_v1",
        "profile_type": "layer_count_proxy_not_wall_clock",
        "candidate_layers": [6, 12, 18, 24],
        "depth_cost_proxy": {
            "12": 0.5,
            "18": 0.75,
            "24": 1.0,
        },
        "depth_savings_proxy_vs_full": {
            "12": 0.5,
            "18": 0.25,
            "24": 0.0,
        },
        "training_started": False,
        "evaluation_started": False,
    }
    latency_payload["latency_profile_identity"] = canonical_json_sha256(latency_payload)

    _write_json(exit_target_path, target_payload)
    _write_json(latency_path, latency_payload)
    _write_jsonl(trace_path, rows)

    manifest = {
        "schema_version": "b3_02_exit_prerequisite_materialization_manifest_v1",
        "accepted_dlcm_identity": chain["accepted_lse_manifest"]["accepted_dlcm_identity"],
        "accepted_lse_identity": chain["accepted_lse_manifest"]["accepted_lse_identity"],
        "b2_phase_final_closure_identity": chain["b2_closure_manifest"]["phase_final_closure_identity"],
        "accepted_lse_checkpoint_sha256": chain["accepted_lse_manifest"]["accepted_lse_checkpoint_sha256"],
        "exit_target_manifest": str(exit_target_path),
        "exit_target_manifest_sha256": sha256_file(exit_target_path),
        "latency_profile": str(latency_path),
        "latency_profile_sha256": sha256_file(latency_path),
        "calibration_trace": str(trace_path),
        "calibration_trace_sha256": sha256_file(trace_path),
        "records": len(rows),
        "counts_by_depth": counts_by_depth,
        "target_exits_by_depth": exits_by_depth,
        "training_started": False,
        "evaluation_started": False,
        "final_content_accessed": False,
        "checkpoint_generated": False,
        "artifact_written": True,
    }
    manifest["materialization_identity"] = canonical_json_sha256(manifest)
    return manifest

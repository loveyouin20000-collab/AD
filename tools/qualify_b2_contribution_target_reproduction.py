#!/usr/bin/env python3
"""Verify, compare, and atomically render deterministic B2-04B evidence."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

JSON_NAME = "b2_04b_contribution_targets_manifest.json"
MARKDOWN_NAME = "b2_04b_contribution_targets_report.md"
STATUS = "deterministic_dual_contribution_target_reproduction"

NEGATIVE_CONTROL_CASE_IDS: tuple[str, ...] = (
    "record_file_byte_drift",
    "record_scientific_hash_drift",
    "coalition_utility_component_drift",
    "raw_utility_drift",
    "centered_value_drift",
    "signed_shapley_drift",
    "allocation_drift",
    "efficiency_residual_above_tolerance",
    "changed_split_membership",
    "training_record_moved_to_calibration",
    "calibration_record_in_gt_fitting",
    "evaluation_record_in_normalization",
    "gt_calibration_statistic_drift",
    "shapley_normalization_statistic_drift",
    "teacher_cache_identity_drift",
    "descriptor_collection_identity_drift",
    "descriptor_record_identity_drift",
    "wrong_split_checkpoint_profile",
    "target_domain_or_visa_source",
    "missing_record",
    "extra_record",
    "orphan_pt",
    "path_traversal",
    "symlink_escape",
    "missing_receipt",
    "receipt_mismatch",
    "output_directory_collision",
    "completed_run_reuse",
    "resume_attempt",
    "wrong_expected_plan_sha",
    "dirty_official_worktree",
    "non_descendant_official_head",
    "moved_or_missing_contract_tag",
    "nonzero_teacher_forward_count",
)

_VALIDATION_KEYS = ("focused_pytest", "full_cpu_pytest", "ruff", "mypy")
_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_.-])/(?:[^\s]+)")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--qualification-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", default="0")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _fail(detail: str) -> None:
    from rad.phase_b.b2_contribution_targets import ContributionTargetError

    raise ContributionTargetError("B2_CONTRIBUTION_QUALIFICATION_INVALID", detail)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be an object")
    return value


def _passed(value: Any, field: str) -> Mapping[str, Any]:
    block = _mapping(value, field)
    if block.get("status") != "passed":
        _fail(f"{field} did not pass")
    return block


def _sanitize_summary(value: Any, field: str) -> str:
    if not isinstance(value, str):
        _fail(f"{field}.summary must be a string")
    return _ABSOLUTE_PATH.sub("<path>", value).strip()


def _load_qualification_results(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail(f"cannot load qualification results: {exc}")
    root = _mapping(raw, "qualification results")
    if root.get("schema_version") != 1:
        _fail("qualification results schema_version must equal 1")

    semantic = _passed(root.get("semantic_spot_checks"), "semantic_spot_checks")
    if (
        semantic.get("sample_count") != 6
        or semantic.get("depths") != [12, 18, 24]
        or semantic.get("run_a_equals_run_b") is not True
    ):
        _fail("semantic spot checks are incomplete")
    source = _passed(root.get("source_only_audit"), "source_only_audit")
    if (
        source.get("target_domain_record_count") != 0
        or source.get("teacher_forward_count") != 0
    ):
        _fail("source-only audit is not clean")
    negative = _passed(root.get("negative_controls"), "negative_controls")
    if (
        negative.get("required") != len(NEGATIVE_CONTROL_CASE_IDS)
        or negative.get("passed") != len(NEGATIVE_CONTROL_CASE_IDS)
        or tuple(negative.get("case_ids", ())) != NEGATIVE_CONTROL_CASE_IDS
    ):
        _fail("negative-control matrix is incomplete or reordered")
    validation = _mapping(root.get("validation"), "validation")
    if set(validation) != set(_VALIDATION_KEYS):
        _fail("validation results must contain focused, full, Ruff, and mypy decisions")
    normalized_validation: dict[str, Any] = {}
    for name in _VALIDATION_KEYS:
        result = _passed(validation.get(name), f"validation.{name}")
        if result.get("exit_code") != 0:
            _fail(f"validation.{name} has a nonzero exit code")
        normalized_validation[name] = {
            "status": "passed",
            "exit_code": 0,
            "summary": _sanitize_summary(result.get("summary"), f"validation.{name}"),
        }
    return {
        "schema_version": 1,
        "semantic_spot_checks": dict(semantic),
        "source_only_audit": dict(source),
        "negative_controls": {
            **dict(negative),
            "case_ids": list(NEGATIVE_CONTROL_CASE_IDS),
        },
        "validation": normalized_validation,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_evidence(verified: Any) -> dict[str, Any]:
    rows = verified.manifest["records"]
    return {
        "contribution_plan_scientific_sha256": verified.manifest[
            "contribution_plan_scientific_sha256"
        ],
        "layered_identities": {
            key: verified.manifest[key]
            for key in __import__(
                "rad.phase_b.b2_contribution_targets", fromlist=["SEVEN_LAYERED_IDENTITY_KEYS"]
            ).SEVEN_LAYERED_IDENTITY_KEYS
        },
        "ordered_record_hashes": [
            {
                "stable_sample_id": row["stable_sample_id"],
                "contribution_target_record_scientific_sha256": row[
                    "contribution_target_record_scientific_sha256"
                ],
            }
            for row in rows
        ],
        "split_counts": dict(verified.manifest["split_counts"]),
        "teacher_forward_count": verified.teacher_forward_count,
        "repository_identity": verified.manifest.get("repository_identity"),
    }


def _build_evidence(
    *,
    config_path: Path,
    config: Any,
    run_a: Any,
    run_b: Any,
    comparison: Any,
    qualification_results: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    comparison_payload = dataclasses.asdict(comparison)
    required_predicates = (
        key
        for key in comparison_payload
        if key not in {"scientifically_equivalent", "reasons", "file_byte_equal"}
    )
    if comparison.scientifically_equivalent is not True or any(
        comparison_payload[key] is not True for key in required_predicates
    ):
        _fail("verified collections are not scientifically equivalent")
    if run_a.teacher_forward_count != 0 or run_b.teacher_forward_count != 0:
        _fail("teacher forward count must remain zero")
    return {
        "schema_version": 1,
        "status": STATUS,
        "seed": seed,
        "configuration": {
            "configuration_id": config.configuration_id,
            "contract_stage": config.contract_stage,
            "configuration_file_sha256": _sha256_file(config_path),
            "expected_contribution_contract_tag": config.expected_contribution_contract_tag,
            "expected_contribution_contract_commit": (
                config.expected_contribution_contract_commit
            ),
        },
        "teacher_forward_count": 0,
        "runs": {
            "run_a": _run_evidence(run_a),
            "run_b": _run_evidence(run_b),
        },
        "comparison": comparison_payload,
        "qualification_results": dict(qualification_results),
        "scope_exclusions": [
            "teacher_or_backbone_forward",
            "target_domain_data",
            "raw_tensors",
            "raw_utility_dumps",
            "timestamps",
            "absolute_paths",
        ],
    }


def _render_json(evidence: Mapping[str, Any]) -> str:
    return json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _render_markdown(evidence: Mapping[str, Any]) -> str:
    comparison = _mapping(evidence["comparison"], "comparison")
    runs = _mapping(evidence["runs"], "runs")
    run_a = _mapping(runs["run_a"], "runs.run_a")
    return (
        "# B2-04B Contribution-Target Reproduction\n\n"
        f"- status: `{evidence['status']}`\n"
        f"- scientifically equivalent: `{str(comparison['scientifically_equivalent']).lower()}`\n"
        f"- teacher forward count: `{evidence['teacher_forward_count']}`\n"
        f"- contribution plan: `{run_a['contribution_plan_scientific_sha256']}`\n"
        f"- verified records per run: `{len(run_a['ordered_record_hashes'])}`\n"
        f"- file-byte equality (diagnostic only): "
        f"`{str(comparison['file_byte_equal']).lower()}`\n\n"
        "Qualification used production disk verification and exact scientific comparison. "
        "Raw tensors, absolute paths, timestamps, and target-domain data are excluded.\n"
    )


def _write_atomic_pair(output_dir: Path, json_text: str, markdown_text: str) -> None:
    json_path = output_dir / JSON_NAME
    markdown_path = output_dir / MARKDOWN_NAME
    if output_dir.exists() and not output_dir.is_dir():
        _fail("output directory path is not a directory")
    if json_path.exists() or json_path.is_symlink() or markdown_path.exists() or markdown_path.is_symlink():
        _fail("qualification output collision")
    created_output_dir = not output_dir.exists()
    output_dir.mkdir(parents=True, exist_ok=True)
    planned = ((json_path, json_text), (markdown_path, markdown_text))
    temporary: list[Path] = []
    replaced: list[Path] = []
    try:
        for destination, content in planned:
            descriptor, name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=output_dir
            )
            temp_path = Path(name)
            temporary.append(temp_path)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        for temp_path, (destination, _content) in zip(tuple(temporary), planned, strict=True):
            os.replace(temp_path, destination)
            temporary.remove(temp_path)
            replaced.append(destination)
    except BaseException:
        # Either both evidence targets appear or none does: roll back every
        # already-replaced destination and drop the temporaries.
        for path in (*temporary, *replaced):
            path.unlink(missing_ok=True)
        if created_output_dir and output_dir.is_dir() and not any(output_dir.iterdir()):
            output_dir.rmdir()
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from rad.phase_b.b2_contribution_targets import (
        ContributionTargetError,
        compare_contribution_target_collections,
        load_contribution_targets_config,
        verify_contribution_target_collection,
    )

    try:
        seed = int(args.seed)
        config_path = Path(args.config)
        config = load_contribution_targets_config(config_path)
        run_a = verify_contribution_target_collection(config=config, run_dir=Path(args.run_a))
        run_b = verify_contribution_target_collection(config=config, run_dir=Path(args.run_b))
        comparison = compare_contribution_target_collections(first=run_a, second=run_b)
        results = _load_qualification_results(Path(args.qualification_results))
        evidence = _build_evidence(
            config_path=config_path,
            config=config,
            run_a=run_a,
            run_b=run_b,
            comparison=comparison,
            qualification_results=results,
            seed=seed,
        )
        json_text = _render_json(evidence)
        markdown_text = _render_markdown(evidence)
        if not args.dry_run:
            _write_atomic_pair(Path(args.output_dir), json_text, markdown_text)
        print(f"status = {STATUS}")
        print("scientifically_equivalent = true")
        print("teacher_forward_count = 0")
        return 0
    except (ContributionTargetError, OSError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

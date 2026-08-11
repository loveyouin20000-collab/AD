#!/usr/bin/env python3
"""Materialize the B3-07 paper results update from frozen evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rad.phase_b import b3_paper_results_update as update  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Close the B3-07 paper results update")
    parser.add_argument(
        "--b2-manifest",
        type=Path,
        default=Path("docs/phase_b/b2_08_paper_results_manifest.json"),
    )
    parser.add_argument(
        "--b3-manifest",
        type=Path,
        default=Path("docs/phase_b/b3_06_early_exit_phase_closure_manifest.json"),
    )
    parser.add_argument(
        "--b4-weight-manifest",
        type=Path,
        default=Path("docs/phase_b/b4_01_dlcm_adaptive_weight_evidence_manifest.json"),
    )
    parser.add_argument(
        "--b4-release-manifest",
        type=Path,
        default=Path("docs/phase_b/b4_02_final_local_paper_release_manifest.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/phase_b"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _tracked_pt_count() -> int:
    output = subprocess.check_output(["git", "ls-files", "*.pt"], cwd=REPO_ROOT, text=True)
    return len([line for line in output.splitlines() if line.strip()])


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _atomic_write(path.with_suffix(path.suffix + ".sha256"), update.sha256_file(path) + "  " + path.name + "\n")


def _write_text(path: Path, text: str) -> None:
    _atomic_write(path, text)
    _atomic_write(path.with_suffix(path.suffix + ".sha256"), update.sha256_file(path) + "  " + path.name + "\n")


def _paper_results_summary(payload: dict[str, Any]) -> str:
    identities = payload["bound_identities"]
    results = payload["result_summary"]
    return f"""# B3-07 Paper Results Update

Status: {payload["status"]}

## Paper Position

| Component | Status | Position |
|---|---|---|
| DLCM dynamic layer fusion | accepted | primary sample-adaptive fusion mechanism |
| LSE layer-sufficiency validation | qualified | supporting validation |
| Early-exit | negative result | limitation and future work; full-depth fallback retained |

## Result Summary

```text
beta_star_decimal = {results["beta_star_decimal"]}
lse_calibration_nll = {results["lse_calibration_nll"]}
lse_required_depths = {results["lse_required_depths"]}
adaptive_weight_calibration_records = {results["adaptive_weight_calibration_records"]}
adaptive_weight_max_sample_linf_delta_from_uniform = {results["adaptive_weight_max_sample_linf_delta_from_uniform"]}
early_exit_candidate_depths = {results["early_exit_candidate_depths"]}
early_exit_fallback_depth = {results["early_exit_fallback_depth"]}
early_exit_positive_signal_count = {results["early_exit_positive_signal_count"]}
```

The accepted system uses DLCM V5 sample-adaptive layer fusion and retains the
qualified LSE validation. The conservative early-exit contract produced no legal
positive exit signal, so it is not an accepted mechanism and the full-depth
fallback remains in force.

## Bound Identities

```text
accepted_dlcm_identity = {identities["accepted_dlcm_identity"]}
v5_deployment_identity = {identities["v5_deployment_identity"]}
accepted_lse_identity = {identities["accepted_lse_identity"]}
b2_phase_final_closure_identity = {identities["b2_phase_final_closure_identity"]}
b3_06_phase_closure_identity = {identities["b3_06_phase_closure_identity"]}
b4_01_weight_evidence_identity = {identities["b4_01_weight_evidence_identity"]}
b4_02_final_release_identity = {identities["b4_02_final_release_identity"]}
update_identity = {payload["update_identity"]}
```
"""


def _evidence_index(payload: dict[str, Any]) -> str:
    documents = "\n".join(f"| `{path}` | Frozen source evidence. |" for path in payload["source_documents"])
    return f"""# B3-07 Paper Evidence Index

Status: {payload["status"]}.

## Claims

| Claim | Position | Bound Evidence |
|---|---|---|
| DLCM V5 has accepted sample-adaptive layer fusion. | main contribution | B2-08, B4-01, B4-02 |
| LSE is qualified layer-sufficiency validation. | supporting validation | B2-08, B4-02 |
| Early-exit is not an accepted mechanism. | negative result and future work | B3-06, B4-02 |
| Full-depth fallback remains required. | accepted behavior | B3-06, B4-02 |

## Source Documents

| File | Role |
|---|---|
{documents}

## Boundary

```text
training_started = false
evaluation_started = false
final_content_accessed = false
model_artifact_generated = false
tracked_pt_files = 0
pushed = false
pr_opened = false
```
"""


def main() -> int:
    args = parse_args()
    payload = update.build_b3_paper_results_update_manifest(
        b2_manifest=update.load_json(_resolve(args.b2_manifest)),
        b3_manifest=update.load_json(_resolve(args.b3_manifest)),
        b4_weight_manifest=update.load_json(_resolve(args.b4_weight_manifest)),
        b4_release_manifest=update.load_json(_resolve(args.b4_release_manifest)),
        tracked_pt_count=_tracked_pt_count(),
    )
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    output_dir = _resolve(args.output_dir)
    _write_json(output_dir / "b3_07_paper_results_update_manifest.json", payload)
    _write_text(output_dir / "b3_07_paper_results_update.md", _paper_results_summary(payload))
    _write_text(output_dir / "b3_07_paper_evidence_index.md", _evidence_index(payload))
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

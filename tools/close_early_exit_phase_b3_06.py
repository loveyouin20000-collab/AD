#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rad.phase_b import b3_early_exit_phase_closure as closure  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Close B3 early-exit phase for paper-ready evidence")
    p.add_argument(
        "--b3-01-evidence",
        type=Path,
        default=Path("docs/phase_b/b3_01_early_exit_preflight_evidence.json"),
    )
    p.add_argument(
        "--b3-02-evidence",
        type=Path,
        default=Path("docs/phase_b/b3_02_exit_prerequisite_materialization_evidence.json"),
    )
    p.add_argument(
        "--b3-03-evidence",
        type=Path,
        default=Path("docs/phase_b/b3_03_exit_policy_training_contract_evidence.json"),
    )
    p.add_argument(
        "--b3-04-evidence",
        type=Path,
        default=Path("docs/phase_b/b3_04_exit_target_positive_signal_contract_evidence.json"),
    )
    p.add_argument(
        "--b3-05-evidence",
        type=Path,
        default=Path("docs/phase_b/b3_05_early_exit_negative_result_evidence.json"),
    )
    p.add_argument("--output-dir", type=Path, default=Path("docs/phase_b"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _tracked_pt_count() -> int:
    out = subprocess.check_output(["git", "ls-files", "*.pt"], cwd=REPO_ROOT).decode()
    return len([line for line in out.splitlines() if line.strip()])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    path.with_suffix(path.suffix + ".sha256").write_text(
        closure.sha256_file(path) + "  " + path.name + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(
        closure.sha256_file(path) + "  " + path.name + "\n",
        encoding="utf-8",
    )


def _paper_summary(payload: dict[str, Any]) -> str:
    row = payload["negative_result_table"][0]
    return f"""# B3-06 Early-Exit Paper Results Summary

Status: {payload["status"]}

## Result Table

| Mechanism | Candidate depths | Positive exit targets | Positive signals | Accepted mechanism | Final behavior |
|---|---:|---:|---:|---|---|
| Early-exit policy | {row["candidate_depths"]} | {row["positive_exit_targets"]} | {row["positive_signal_count"]} | no | {payload["accepted_system_behavior"]} |

## Paper-Ready Result

Early-exit was evaluated as a downstream efficiency extension after the DLCM
and LSE artifacts had been accepted. Under the conservative positive-signal
contract, neither the materialized exit targets nor the LSE-derived calibration
trace produced a legal positive early-exit signal at depths 12 or 18. Therefore
early-exit policy training was not unlocked, no early-exit checkpoint was
generated, and the accepted system retains full-depth inference.

## Interpretation

This result does not remove the paper's dynamic fusion contribution. DLCM and
LSE remain the accepted mechanisms; early-exit is reported as a negative result
and future-work direction under the current conservative gate.

## Bound Identities

```text
accepted_dlcm_identity = {payload["primary_identities"]["accepted_dlcm_identity"]}
accepted_lse_identity = {payload["primary_identities"]["accepted_lse_identity"]}
b2_phase_final_closure_identity = {payload["primary_identities"]["b2_phase_final_closure_identity"]}
b3_05_line_closure_identity = {payload["phase_identities"]["b3_05_line_closure_identity"]}
b3_06_phase_closure_identity = {payload["phase_closure_identity"]}
```

## Boundary

```text
training_started_in_b3_06 = false
evaluation_started_in_b3_06 = false
final_content_accessed_in_b3_06 = false
model_artifact_generated_in_b3_06 = false
tracked_pt_files = {payload["boundary"]["tracked_pt_files"]}
```
"""


def _evidence_index(payload: dict[str, Any]) -> str:
    docs = "\n".join(f"| `{doc}` | B3 early-exit evidence. |" for doc in payload["evidence_documents"])
    return f"""# B3-06 Early-Exit Evidence Index

Status: {payload["status"]}.

This index maps paper-facing early-exit claims to tracked evidence. Ignored
runtime artifacts remain referenced only through identities and SHA256 values.

## Claims

| Claim | Evidence |
|---|---|
| Early-exit was wired behind the accepted DLCM/LSE chain. | `docs/phase_b/b3_01_early_exit_preflight_evidence.json` |
| Exit targets, latency proxy, and calibration trace were materialized before any training. | `docs/phase_b/b3_02_exit_prerequisite_materialization_evidence.json` |
| No positive exit targets were available under the first target materialization. | `docs/phase_b/b3_03_exit_policy_training_contract_evidence.json` |
| The conservative positive-signal redefinition still produced zero legal positives. | `docs/phase_b/b3_04_exit_target_positive_signal_contract_evidence.json` |
| The early-exit line is closed as a negative result with full-depth fallback. | `docs/phase_b/b3_05_early_exit_negative_result_evidence.json`; `docs/phase_b/b3_06_early_exit_phase_closure_manifest.json` |
| Dynamic fusion and LSE are not abandoned by the early-exit negative result. | `docs/phase_b/b3_06_early_exit_paper_results_summary.md` |

## Evidence Files

| File | Purpose |
|---|---|
{docs}

## Boundary

```text
B3-06 does not start training.
B3-06 does not start evaluation.
B3-06 does not read Final content.
B3-06 does not generate model artifacts.
B3-06 does not push or open a PR.
```
"""


def _negative_table_csv(payload: dict[str, Any]) -> str:
    row = payload["negative_result_table"][0]
    return "\n".join(
        [
            "mechanism,candidate_depths,fallback_depth,records,positive_exit_targets,positive_signal_count,accepted_as_final_mechanism,paper_interpretation",
            (
                f"{row['component']},\"{row['candidate_depths']}\",{row['fallback_depth']},"
                f"{row['records']},{row['positive_exit_targets']},{row['positive_signal_count']},"
                f"{str(row['accepted_as_final_mechanism']).lower()},{row['paper_interpretation']}"
            ),
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    payload = closure.build_early_exit_phase_closure(
        b3_01_evidence=closure.load_json(_resolve(args.b3_01_evidence)),
        b3_02_evidence=closure.load_json(_resolve(args.b3_02_evidence)),
        b3_03_evidence=closure.load_json(_resolve(args.b3_03_evidence)),
        b3_04_evidence=closure.load_json(_resolve(args.b3_04_evidence)),
        b3_05_evidence=closure.load_json(_resolve(args.b3_05_evidence)),
        tracked_pt_count=_tracked_pt_count(),
    )
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    output_dir = _resolve(args.output_dir)
    _write_json(output_dir / "b3_06_early_exit_phase_closure_manifest.json", payload)
    _write_text(output_dir / "b3_06_early_exit_paper_results_summary.md", _paper_summary(payload))
    _write_text(output_dir / "b3_06_early_exit_evidence_index.md", _evidence_index(payload))
    _write_text(output_dir / "b3_06_early_exit_negative_result_table.csv", _negative_table_csv(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

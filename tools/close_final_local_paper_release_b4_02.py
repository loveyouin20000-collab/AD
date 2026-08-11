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

from rad.phase_b import b4_final_paper_release as release  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Close final local paper release")
    p.add_argument("--b2-manifest", type=Path, default=Path("docs/phase_b/b2_08_paper_results_manifest.json"))
    p.add_argument("--b3-manifest", type=Path, default=Path("docs/phase_b/b3_06_early_exit_phase_closure_manifest.json"))
    p.add_argument("--b4-weight-manifest", type=Path, default=Path("docs/phase_b/b4_01_dlcm_adaptive_weight_evidence_manifest.json"))
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
        release.sha256_file(path) + "  " + path.name + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(
        release.sha256_file(path) + "  " + path.name + "\n",
        encoding="utf-8",
    )


def _results_summary(payload: dict[str, Any]) -> str:
    ids = payload["bound_identities"]
    adapt = payload["adaptive_weight_summary"]
    return f"""# B4-02 Final Paper Results Summary

Status: {payload["status"]}

## Final Paper Claims

| Component | Status | Paper Position |
|---|---|---|
| DLCM sample-adaptive layer fusion | accepted | main contribution |
| LSE layer-sufficiency validation | qualified | supporting validation |
| Early-exit | negative result | limitation / future work |

## Core Result

The final local release supports the paper's main claim: VirtualAD-style fixed
equal fusion is replaced by an accepted DLCM V5 deployment with sample-adaptive
layer weights. LSE is qualified as supporting validation. Early-exit was explored
under accepted DLCM/LSE identities but remains a negative result under the
conservative gate, so the accepted system retains full-depth fallback.

## Adaptive Weight Evidence

```text
sample_adaptive_variation_observed = {str(adapt["sample_adaptive_variation_observed"]).lower()}
uniform_equivalent_at_tolerance = {str(adapt["uniform_equivalent_at_tolerance"]).lower()}
calibration_records = {adapt["calibration_records"]}
deployment_max_sample_linf_delta_from_uniform = {adapt["deployment_max_sample_linf_delta_from_uniform"]}
```

## Bound Identities

```text
accepted_dlcm_identity = {ids["accepted_dlcm_identity"]}
v5_deployment_identity = {ids["v5_deployment_identity"]}
accepted_lse_identity = {ids["accepted_lse_identity"]}
b2_phase_final_closure_identity = {ids["b2_phase_final_closure_identity"]}
b3_06_phase_closure_identity = {ids["b3_06_phase_closure_identity"]}
b4_01_weight_evidence_identity = {ids["b4_01_weight_evidence_identity"]}
final_release_identity = {payload["final_release_identity"]}
```
"""


def _evidence_index(payload: dict[str, Any]) -> str:
    docs = "\n".join(
        f"| `{doc}` | Final local release evidence. |" for doc in payload["evidence_documents"]
    )
    return f"""# B4-02 Global Paper Evidence Index

Status: {payload["status"]}.

This index is the local release map for paper-facing claims. It consolidates
B2 accepted DLCM/LSE evidence, B3 early-exit negative evidence, and B4 adaptive
weight evidence.

## Claims

| Claim | Evidence |
|---|---|
| DLCM V5 is the accepted dynamic layer fusion artifact. | `docs/phase_b/b2_08_paper_results_manifest.json`; `docs/phase_b/b2_07_phase_final_closure_manifest.json` |
| Accepted V5 deployment weights show sample-adaptive variation. | `docs/phase_b/b4_01_dlcm_adaptive_weight_evidence_manifest.json`; `docs/phase_b/b4_01_dlcm_adaptive_weight_trace.csv` |
| LSE is qualified as supporting validation. | `docs/phase_b/b2_06e_lse_qualification_decision_manifest.json`; `docs/phase_b/b2_08_paper_results_manifest.json` |
| Early-exit is a negative result, not an accepted mechanism. | `docs/phase_b/b3_06_early_exit_phase_closure_manifest.json` |
| No new training, evaluation, Final access, model artifact, push, or PR occurred in final release closure. | `docs/phase_b/b4_02_final_local_paper_release_manifest.json`; `docs/phase_b/b4_02_release_checklist.md` |

## Evidence Files

| File | Purpose |
|---|---|
{docs}
"""


def _checklist(payload: dict[str, Any]) -> str:
    b = payload["boundary"]
    return f"""# B4-02 Release Checklist

```text
release_decision = {payload["release_decision"]}
final_release_identity = {payload["final_release_identity"]}
```

## Boundary

```text
training_started_in_release = {str(b["training_started_in_release"]).lower()}
evaluation_started_in_release = {str(b["evaluation_started_in_release"]).lower()}
final_content_accessed_in_release = {str(b["final_content_accessed_in_release"]).lower()}
model_artifact_generated_in_release = {str(b["model_artifact_generated_in_release"]).lower()}
tracked_pt_files = {b["tracked_pt_files"]}
pushed = {str(b["pushed"]).lower()}
pr_opened = {str(b["pr_opened"]).lower()}
```

## Next Decision

The local release is ready for a separate push/PR decision. No remote action is
included in this closure.
"""


def main() -> int:
    args = parse_args()
    payload = release.build_final_paper_release_manifest(
        b2_manifest=release.load_json(_resolve(args.b2_manifest)),
        b3_manifest=release.load_json(_resolve(args.b3_manifest)),
        b4_weight_manifest=release.load_json(_resolve(args.b4_weight_manifest)),
        tracked_pt_count=_tracked_pt_count(),
    )
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    output_dir = _resolve(args.output_dir)
    _write_json(output_dir / "b4_02_final_local_paper_release_manifest.json", payload)
    _write_text(output_dir / "b4_02_final_paper_results_summary.md", _results_summary(payload))
    _write_text(output_dir / "b4_02_global_paper_evidence_index.md", _evidence_index(payload))
    _write_text(output_dir / "b4_02_release_checklist.md", _checklist(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

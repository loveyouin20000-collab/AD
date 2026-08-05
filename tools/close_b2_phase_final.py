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

from rad.phase_b import b2_phase_final_closure as closure  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Close B2 phase final local handoff")
    p.add_argument("--output-dir", type=Path, default=Path("docs/phase_b"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _git_sha() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
        .decode()
        .strip()
    )


def _tracked_pt_count() -> int:
    out = subprocess.check_output(["git", "ls-files", "*.pt"], cwd=REPO_ROOT).decode()
    return len([line for line in out.splitlines() if line.strip()])


def _write_text_with_sha(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(
        closure.sha256_file(path) + "  " + path.name + "\n",
        encoding="utf-8",
    )


def _write_json_with_sha(path: Path, payload: dict[str, Any]) -> None:
    _write_text_with_sha(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _report(manifest: dict[str, Any]) -> str:
    return f"""# B2 Phase Final Closure Report

Status: {manifest["status"]}

Phase final closure identity:

```text
{manifest["phase_final_closure_identity"]}
```

Accepted identities:

```text
accepted_dlcm_identity = {manifest["accepted_dlcm_identity"]}
v5_deployment_identity = {manifest["v5_deployment_identity"]}
accepted_lse_identity = {manifest["accepted_lse_identity"]}
H_lse_qualification = {manifest["H_lse_qualification"]}
```

Frozen artifact hashes:

```text
accepted_v5_checkpoint_sha256 = {manifest["accepted_v5_checkpoint_sha256"]}
accepted_lse_checkpoint_sha256 = {manifest["accepted_lse_checkpoint_sha256"]}
selector_signal_layout_hash = {manifest["selector_signal_layout_hash"]}
```

LSE qualification:

```text
calibration_nll = {manifest["calibration_nll"]}
max_calibration_nll = {manifest["max_calibration_nll"]}
evaluated_rows = {manifest["evaluated_rows"]}
```

Boundary:

```text
training_started_in_b2_07 = {str(manifest["training_started_in_b2_07"]).lower()}
evaluation_started_in_b2_07 = {str(manifest["evaluation_started_in_b2_07"]).lower()}
final_content_accessed_in_b2_07 = {str(manifest["final_content_accessed_in_b2_07"]).lower()}
tracked_pt_count = {manifest["tracked_pt_count"]}
pushed = {str(manifest["pushed"]).lower()}
pr_opened = {str(manifest["pr_opened"]).lower()}
```
"""


def main() -> int:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    manifest_path = output_dir / "b2_07_phase_final_closure_manifest.json"
    report_path = output_dir / "b2_07_phase_final_closure_report.md"

    manifest = closure.build_phase_final_closure_manifest(
        accepted_gate_evidence=closure.load_json(
            REPO_ROOT / "docs/phase_b/b2_06a_lse_accepted_gate_preflight_evidence.json"
        ),
        reference_packaging_evidence=closure.load_json(
            REPO_ROOT / "docs/phase_b/b2_06b_accepted_v5_reference_packaging_evidence.json"
        ),
        prerequisite_evidence=closure.load_json(
            REPO_ROOT / "docs/phase_b/b2_06c_lse_prerequisite_materialization_evidence.json"
        ),
        training_evidence=closure.load_json(
            REPO_ROOT / "docs/phase_b/b2_06d_lse_first_controlled_run_evidence.json"
        ),
        qualification_decision=closure.load_json(
            REPO_ROOT / "docs/phase_b/b2_06e_lse_qualification_decision_manifest.json"
        ),
        accepted_lse_manifest=closure.load_json(
            REPO_ROOT / "docs/phase_b/b2_06f_accepted_lse_manifest.json"
        ),
        accepted_lse_receipt=closure.load_json(
            REPO_ROOT / "docs/phase_b/b2_06f_accepted_lse_closure_receipt.json"
        ),
        accepted_lse_evidence=closure.load_json(
            REPO_ROOT / "docs/phase_b/b2_06f_accepted_lse_closure_evidence.json"
        ),
        git_sha=_git_sha(),
        tracked_pt_count=_tracked_pt_count(),
        pushed=False,
        pr_opened=False,
    )
    if args.dry_run:
        print(json.dumps({"ready": True, **manifest}, indent=2, sort_keys=True))
        return 0
    if manifest_path.exists() or report_path.exists():
        raise SystemExit("B2_PHASE_FINAL_CLOSURE_ALREADY_EXISTS")
    _write_json_with_sha(manifest_path, manifest)
    _write_text_with_sha(report_path, _report(manifest))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

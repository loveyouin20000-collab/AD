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

from rad.phase_b import b3_exit_training_contract as contract  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Close B3 exit policy training contract")
    p.add_argument(
        "--prerequisite-manifest",
        type=Path,
        default=Path("docs/phase_b/b3_02_exit_prerequisite_materialization_manifest.json"),
    )
    p.add_argument("--output-dir", type=Path, default=Path("docs/phase_b"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def _tracked_pt_count() -> int:
    out = subprocess.check_output(["git", "ls-files", "*.pt"], cwd=REPO_ROOT).decode()
    return len([line for line in out.splitlines() if line.strip()])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(
        contract.sha256_file(path) + "  " + path.name + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(
        contract.sha256_file(path) + "  " + path.name + "\n",
        encoding="utf-8",
    )


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# B3-03 Exit Policy Training Contract

Status: {payload["decision"]}

Reason:

```text
{payload["reason"]}
```

Training contract identity:

```text
{payload["training_contract_identity"]}
```

Target distribution:

```text
positive_exit_targets = {payload["positive_exit_targets"]}
depth 12 exits = {payload["target_exits_by_depth"].get("12")}
depth 18 exits = {payload["target_exits_by_depth"].get("18")}
```

Boundary:

```text
training_unlocked = {str(payload["training_unlocked"]).lower()}
training_started = {str(payload["training_started"]).lower()}
evaluation_started = {str(payload["evaluation_started"]).lower()}
final_content_accessed = {str(payload["final_content_accessed"]).lower()}
checkpoint_generated = {str(payload["checkpoint_generated"]).lower()}
fallback_depth = {payload["fallback_depth"]}
tracked_pt_count = {payload["tracked_pt_count"]}
```
"""


def main() -> int:
    args = parse_args()
    prereq_path = args.prerequisite_manifest if args.prerequisite_manifest.is_absolute() else REPO_ROOT / args.prerequisite_manifest
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    payload = contract.build_exit_training_contract(
        prerequisite_manifest=contract.load_json(prereq_path),
        git_sha=_git_sha(),
        tracked_pt_count=_tracked_pt_count(),
    )
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    _write_json(output_dir / "b3_03_exit_policy_training_contract.json", payload)
    _write_text(output_dir / "b3_03_exit_policy_training_contract.md", _markdown(payload))
    evidence = {
        "schema_version": "b3_03_exit_policy_training_contract_evidence_v1",
        "status": payload["decision"],
        "reason": payload["reason"],
        "training_contract_identity": payload["training_contract_identity"],
        "positive_exit_targets": payload["positive_exit_targets"],
        "training_unlocked": payload["training_unlocked"],
        "training_started": payload["training_started"],
        "evaluation_started": payload["evaluation_started"],
        "checkpoint_generated": payload["checkpoint_generated"],
        "tracked_pt_count": payload["tracked_pt_count"],
    }
    _write_json(output_dir / "b3_03_exit_policy_training_contract_evidence.json", evidence)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

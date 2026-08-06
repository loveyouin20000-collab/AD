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

from rad.phase_b import b3_early_exit_line_closure as closure  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Close B3 early-exit line as negative evidence")
    p.add_argument(
        "--b3-02-manifest",
        type=Path,
        default=Path("docs/phase_b/b3_02_exit_prerequisite_materialization_manifest.json"),
    )
    p.add_argument(
        "--b3-03-contract",
        type=Path,
        default=Path("docs/phase_b/b3_03_exit_policy_training_contract.json"),
    )
    p.add_argument(
        "--b3-04-contract",
        type=Path,
        default=Path("docs/phase_b/b3_04_exit_target_positive_signal_contract.json"),
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


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# B3-05 Early-Exit Line Closure / Negative Result Evidence

Status:

```text
{payload["status"]}
```

Decision:

```text
{payload["decision"]}
```

Reason:

```text
{payload["reason"]}
```

Line closure identity:

```text
{payload["line_closure_identity"]}
```

Bound identities:

```text
accepted_dlcm_identity = {payload["accepted_dlcm_identity"]}
accepted_lse_identity = {payload["accepted_lse_identity"]}
b2_phase_final_closure_identity = {payload["b2_phase_final_closure_identity"]}
b3_02_materialization_identity = {payload["b3_02_materialization_identity"]}
b3_03_training_contract_identity = {payload["b3_03_training_contract_identity"]}
b3_04_positive_signal_contract_identity = {payload["b3_04_positive_signal_contract_identity"]}
```

Negative-result evidence:

```text
positive_exit_targets = {payload["positive_exit_targets"]}
positive_signal_count = {payload["positive_signal_count"]}
depth 12 target exits = {payload["target_exits_by_depth"].get("12")}
depth 18 target exits = {payload["target_exits_by_depth"].get("18")}
depth 12 positive signals = {payload["candidate_positive_counts_by_depth"].get("12")}
depth 18 positive signals = {payload["candidate_positive_counts_by_depth"].get("18")}
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
    payload = closure.build_early_exit_line_closure(
        b3_02_manifest=closure.load_json(_resolve(args.b3_02_manifest)),
        b3_03_training_contract=closure.load_json(_resolve(args.b3_03_contract)),
        b3_04_positive_signal_contract=closure.load_json(_resolve(args.b3_04_contract)),
        tracked_pt_count=_tracked_pt_count(),
    )
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    output_dir = _resolve(args.output_dir)
    _write_json(output_dir / "b3_05_early_exit_line_closure_manifest.json", payload)
    _write_text(output_dir / "b3_05_early_exit_line_closure_report.md", _markdown(payload))
    evidence = {
        "schema_version": "b3_05_early_exit_negative_result_evidence_v1",
        "status": payload["status"],
        "decision": payload["decision"],
        "reason": payload["reason"],
        "line_closure_identity": payload["line_closure_identity"],
        "positive_exit_targets": payload["positive_exit_targets"],
        "positive_signal_count": payload["positive_signal_count"],
        "target_exits_by_depth": payload["target_exits_by_depth"],
        "candidate_positive_counts_by_depth": payload["candidate_positive_counts_by_depth"],
        "training_unlocked": payload["training_unlocked"],
        "training_started": payload["training_started"],
        "evaluation_started": payload["evaluation_started"],
        "checkpoint_generated": payload["checkpoint_generated"],
        "tracked_pt_count": payload["tracked_pt_count"],
    }
    _write_json(output_dir / "b3_05_early_exit_negative_result_evidence.json", evidence)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

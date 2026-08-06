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

from rad.phase_b import b3_exit_target_redefinition as redefine  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Close B3 exit target positive-signal contract")
    p.add_argument(
        "--b3-02-manifest",
        type=Path,
        default=Path("docs/phase_b/b3_02_exit_prerequisite_materialization_manifest.json"),
    )
    p.add_argument("--max-predicted-remaining-gain", type=float, default=0.10)
    p.add_argument("--min-predicted-sufficiency-probability", type=float, default=0.50)
    p.add_argument("--output-dir", type=Path, default=Path("docs/phase_b"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _tracked_pt_count() -> int:
    out = subprocess.check_output(["git", "ls-files", "*.pt"], cwd=REPO_ROOT).decode()
    return len([line for line in out.splitlines() if line.strip()])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(
        redefine.sha256_file(path) + "  " + path.name + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(
        redefine.sha256_file(path) + "  " + path.name + "\n",
        encoding="utf-8",
    )


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# B3-04 Exit Target Positive-Signal Contract

Status: {payload["decision"]}

Positive-signal contract identity:

```text
{payload["positive_signal_contract_identity"]}
```

Thresholds:

```text
max_predicted_remaining_gain = {payload["max_predicted_remaining_gain"]}
min_predicted_sufficiency_probability = {payload["min_predicted_sufficiency_probability"]}
```

Candidate positives:

```text
positive_signal_count = {payload["positive_signal_count"]}
depth 12 = {payload["candidate_positive_counts_by_depth"]["12"]}
depth 18 = {payload["candidate_positive_counts_by_depth"]["18"]}
```

Boundary:

```text
training_unlocked = {str(payload["training_unlocked"]).lower()}
training_started = {str(payload["training_started"]).lower()}
evaluation_started = {str(payload["evaluation_started"]).lower()}
final_content_accessed = {str(payload["final_content_accessed"]).lower()}
checkpoint_generated = {str(payload["checkpoint_generated"]).lower()}
tracked_pt_count = {payload["tracked_pt_count"]}
```
"""


def main() -> int:
    args = parse_args()
    manifest_path = args.b3_02_manifest if args.b3_02_manifest.is_absolute() else REPO_ROOT / args.b3_02_manifest
    manifest = redefine.load_json(manifest_path)
    trace = redefine.load_jsonl(manifest["calibration_trace"])
    latency = redefine.load_json(manifest["latency_profile"])
    payload = redefine.build_positive_signal_contract(
        calibration_trace_rows=trace,
        latency_profile=latency,
        max_predicted_remaining_gain=args.max_predicted_remaining_gain,
        min_predicted_sufficiency_probability=args.min_predicted_sufficiency_probability,
        accepted_lse_identity=manifest["accepted_lse_identity"],
        b3_02_materialization_identity=manifest["materialization_identity"],
        tracked_pt_count=_tracked_pt_count(),
    )
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    _write_json(output_dir / "b3_04_exit_target_positive_signal_contract.json", payload)
    _write_text(output_dir / "b3_04_exit_target_positive_signal_contract.md", _markdown(payload))
    evidence = {
        "schema_version": "b3_04_exit_target_positive_signal_contract_evidence_v1",
        "decision": payload["decision"],
        "positive_signal_contract_identity": payload["positive_signal_contract_identity"],
        "positive_signal_count": payload["positive_signal_count"],
        "candidate_positive_counts_by_depth": payload["candidate_positive_counts_by_depth"],
        "training_unlocked": payload["training_unlocked"],
        "training_started": payload["training_started"],
        "evaluation_started": payload["evaluation_started"],
        "checkpoint_generated": payload["checkpoint_generated"],
        "tracked_pt_count": payload["tracked_pt_count"],
    }
    _write_json(output_dir / "b3_04_exit_target_positive_signal_contract_evidence.json", evidence)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""V5 Final materialization CLI.

C4C supports dry-run plan validation only. Real Final content remains guarded
until a later stage supplies a valid unlock.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rad.phase_b import b2_dlcm_v5_final_unlock as final_unlock  # noqa: E402
from rad.phase_b import b2_dlcm_v5_protocol as protocol  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="B2 DLCM V5 Final materialization tooling.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--expected-plan-sha256", default=None)
    return parser


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=_REPO_ROOT, text=True).strip()


def _production_tooling_diff_since(tag: str) -> list[str]:
    allowed_prefixes = (
        "configs/phase_b/b2_dlcm_v5_final_execution_official_v1.json",
        "docs/",
    )
    output = _git("diff", "--name-only", f"{tag}..HEAD")
    changed = [line for line in output.splitlines() if line]
    return [path for path in changed if not path.startswith(allowed_prefixes)]


def _repo_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    tag = str(config["tooling_baseline_tag"])
    head = _git("rev-parse", "HEAD")
    tag_commit = _git("rev-parse", f"{tag}^{{commit}}")
    descendant = subprocess.run(
        ["git", "merge-base", "--is-ancestor", tag_commit, "HEAD"],
        cwd=_REPO_ROOT,
        check=False,
    ).returncode == 0
    dirty = bool(_git("status", "--short"))
    production_changes = _production_tooling_diff_since(tag)
    return {
        "head": head,
        "tooling_baseline_commit": tag_commit,
        "tooling_baseline_tag": tag,
        "head_is_descendant_of_tooling_tag": descendant,
        "production_tooling_diff_since_tag_empty": not production_changes,
        "worktree_clean": not dirty,
    }


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        protocol.forbid_final_content_access(unlocked=False, context="materialize_v5_no_unlock")
        raise AssertionError("unreachable after fail-closed guard")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.dry_run:
        protocol.forbid_final_content_access(unlocked=False, context="materialize_v5_requires_unlock")
    config = _load_config(args.config)
    if str(config.get("tooling_baseline_commit")) == "TO_BE_REPLACED_BY_V2_TAG_TARGET":
        tag_commit = _git("rev-parse", f"{config['tooling_baseline_tag']}^{{commit}}")
        config = {**config, "tooling_baseline_commit": tag_commit}
    repo_identity = _repo_identity(config)
    final_unlock.validate_repository_gate(repo_identity=repo_identity)
    plan = final_unlock.build_final_execution_plan(
        config=config,
        repo_identity=repo_identity,
    )
    plan_sha = final_unlock.final_execution_plan_sha256(plan)
    expected_plan_sha = args.expected_plan_sha256 or config.get("expected_accepted_v5_final_execution_plan_sha256")
    if expected_plan_sha is not None and expected_plan_sha != plan_sha:
        raise final_unlock.B2DLCMV5FinalUnlockError(
            "B2_DLCM_FINAL_EXECUTION_PLAN_MISMATCH",
            "CLI SHA != recomputed Final execution plan SHA",
        )
    print(json.dumps(final_unlock.dry_run_status(plan_sha256=plan_sha), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (protocol.B2DLCMV5ProtocolError, final_unlock.B2DLCMV5FinalUnlockError) as exc:
        print(f"ERROR {exc.code}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

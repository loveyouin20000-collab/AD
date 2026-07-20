from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_ls_files(prefix: str) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", prefix],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return [line for line in proc.stdout.splitlines() if line.strip()]


def test_no_tracked_files_under_artifacts() -> None:
    """Generated run outputs must stay untracked under artifacts/."""
    tracked = _git_ls_files("artifacts")
    assert tracked == [], (
        "artifacts/ must not contain tracked files; "
        f"found {len(tracked)}: {tracked[:5]}"
    )

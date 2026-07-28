"""Focused repository-identity contract tests for the B2 foundation."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tools import create_b2_tiny_split as subject

EXPECTED_B1_TAG = "b1-strict-independent-v1"
EXPECTED_B1_COMMIT = "3a751b2784a50eb0a08ed49e1db2df0b53608ccc"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    marker = repo / "history.txt"
    previous = marker.read_text(encoding="utf-8") if marker.exists() else ""
    marker.write_text(previous + message + "\n", encoding="utf-8")
    _git(repo, "add", "history.txt")
    _git(
        repo,
        "-c",
        "user.name=B2 Contract Test",
        "-c",
        "user.email=b2-contract@example.invalid",
        "commit",
        "-m",
        message,
    )
    return _git(repo, "rev-parse", "HEAD")


def _repository(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    parent = _commit(repo, "parent")
    base = _commit(repo, "accepted B1 base")
    _git(repo, "tag", EXPECTED_B1_TAG, base)
    return repo, parent, base


def _specification(base: str) -> dict[str, Any]:
    return {"b1_base": {"tag": EXPECTED_B1_TAG, "commit": base}}


def _derive(repo: Path, base: str, *, require_clean: bool = False) -> dict[str, Any]:
    with patch.object(subject, "EXPECTED_B1_COMMIT", base):
        return subject._derive_repository_identity(
            repo,
            _specification(base),
            require_clean=require_clean,
        )


def test_head_equal_to_b1_commit_passes(tmp_path: Path) -> None:
    repo, _, base = _repository(tmp_path)

    identity = _derive(repo, base, require_clean=True)

    assert identity["b1_base_commit"] == base
    assert identity["generation_git_commit"] == base
    assert identity["worktree_clean"] is True


def test_clean_descendant_commit_passes(tmp_path: Path) -> None:
    repo, _, base = _repository(tmp_path)
    descendant = _commit(repo, "B2 descendant")

    identity = _derive(repo, base, require_clean=True)

    assert identity["generation_git_commit"] == descendant


def test_multiple_descendant_commits_pass(tmp_path: Path) -> None:
    repo, _, base = _repository(tmp_path)
    _commit(repo, "B2 first descendant")
    descendant = _commit(repo, "B2 second descendant")

    identity = _derive(repo, base, require_clean=True)

    assert identity["generation_git_commit"] == descendant


def test_sibling_commit_fails(tmp_path: Path) -> None:
    repo, parent, base = _repository(tmp_path)
    _git(repo, "checkout", "-b", "sibling", parent)
    _commit(repo, "sibling of B1")

    with pytest.raises(subject.B2TinySplitCLIError, match="B2_REPOSITORY_ANCESTRY"):
        _derive(repo, base)


def test_unrelated_history_fails(tmp_path: Path) -> None:
    repo, _, base = _repository(tmp_path)
    _git(repo, "checkout", "--orphan", "unrelated")
    _git(repo, "rm", "-rf", ".")
    _commit(repo, "unrelated root")

    with pytest.raises(subject.B2TinySplitCLIError, match="B2_REPOSITORY_ANCESTRY"):
        _derive(repo, base)


def test_moved_b1_tag_fails(tmp_path: Path) -> None:
    repo, _, base = _repository(tmp_path)
    wrong = _commit(repo, "wrong tag target")
    _git(repo, "tag", "-f", EXPECTED_B1_TAG, wrong)

    with pytest.raises(subject.B2TinySplitCLIError, match="B2_REPOSITORY_IDENTITY"):
        _derive(repo, base)


def test_manifest_generation_sha_differing_from_observed_head_fails(
    tmp_path: Path,
) -> None:
    repo, _, base = _repository(tmp_path)
    identity = _derive(repo, base)
    identity["generation_git_commit"] = "0" * 40

    with (
        patch.object(subject, "EXPECTED_B1_COMMIT", base),
        pytest.raises(subject.B2TinySplitCLIError, match="B2_REPOSITORY_IDENTITY"),
    ):
        subject._validate_repository_identity(
            identity,
            observed_head=_git(repo, "rev-parse", "HEAD"),
        )


def test_dirty_official_run_worktree_fails(tmp_path: Path) -> None:
    repo, _, base = _repository(tmp_path)
    (repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(subject.B2TinySplitCLIError, match="B2_WORKTREE_DIRTY"):
        _derive(repo, base, require_clean=True)


def test_temporary_repo_head_passes_as_b1_descendant(tmp_path: Path) -> None:
    """Portable stand-in for real-checkout ancestry (CI has fetch-depth:1, no tags)."""

    repo, _, base = _repository(tmp_path)
    descendant = _commit(repo, "B2-01 descendant stand-in")
    subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", base, descendant],
        check=True,
    )
    identity = _derive(repo, base, require_clean=True)
    assert identity["b1_base_commit"] == base
    assert identity["generation_git_commit"] == descendant


def test_cpu_suite_does_not_require_real_b1_release_tag() -> None:
    """Unit suite must remain hermetic even when local release tags are absent."""

    probe = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", f"{EXPECTED_B1_TAG}^{{commit}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    # Either the tag exists locally (dev worktree) or it does not (CI shallow).
    # In both cases the portable suite above covers the contract without depending
    # on this probe succeeding.
    assert probe.returncode in {0, 128}

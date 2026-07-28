"""RED/guards: B2 CPU qualification must be hermetic on shallow no-tag CI checkouts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from rad.phase_b import b2_teacher_cache as cache
from tests.rad import b2_hermetic as hermetic
from tools import create_b2_teacher_cache as teacher_cli
from tools import create_b2_tiny_split as tiny_cli

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_B1_TAG = "b1-strict-independent-v1"
PRODUCTION_SPLIT_V2 = hermetic.EXPECTED_SPLIT_V2
PRODUCTION_CHECKPOINT = hermetic.EXPECTED_CHECKPOINT_SHA256


def test_repository_identity_suite_uses_temporary_repos_only() -> None:
    identity_tests = (
        REPO_ROOT / "tests" / "rad" / "test_b2_repository_identity.py"
    ).read_text(encoding="utf-8")
    assert "test_b2_01_head_passes_as_b1_descendant" not in identity_tests
    assert "test_current_head_passes_as_b1_descendant" not in identity_tests
    assert "CURRENT_B2_HEAD" not in identity_tests
    assert "temporary_repo_head_passes_as_b1_descendant" in identity_tests
    assert "cpu_suite_does_not_require_real_b1_release_tag" in identity_tests


def test_cpu_suite_does_not_require_real_b1_tag_in_checkout() -> None:
    probe = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", f"{EXPECTED_B1_TAG}^{{commit}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode in {0, 128}


def test_teacher_cache_cli_module_avoids_autodl_path_literals() -> None:
    text = (REPO_ROOT / "tests" / "rad" / "test_b2_teacher_cache_cli.py").read_text(
        encoding="utf-8"
    )
    assert "/root/autodl-tmp/" not in text
    assert "write_b2_split_fixture" in text
    assert "write_hermetic_checkpoint" in text


def test_tiny_split_cli_defaults_to_controlled_tmp_sources() -> None:
    text = (REPO_ROOT / "tests" / "rad" / "test_b2_tiny_split_cli.py").read_text(
        encoding="utf-8"
    )
    assert "populate_controlled_mvtec" in text
    assert 'tmp_path / "controlled_mvtec"' in text
    assert 'source_root or (tmp_path / "controlled_mvtec")' in text
    # Official V2 anchoring may reference production MVTec, but only behind skipif.
    assert "test_official_mvtec_dry_run_reproduces_approved_scientific_v2" in text
    assert 'reason="official Gate-C V2 regression requires production MVTec root"' in text


def test_removing_autodl_tmp_from_environment_does_not_break_hermetic_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AUTODL_TMP", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    split = hermetic.write_b2_split_fixture(tmp_path / "split.json")
    ckpt, digest = hermetic.write_hermetic_checkpoint(tmp_path / "ckpt.pth")
    assert split.is_file()
    assert ckpt.is_file()
    assert digest == hashlib.sha256(ckpt.read_bytes()).hexdigest()
    assert digest != PRODUCTION_CHECKPOINT
    mvtec = hermetic.populate_controlled_mvtec(tmp_path / "mvtec")
    assert any(mvtec.rglob("*.png"))
    roots = hermetic.populate_b1_task_level_roots(tmp_path / "b1")
    assert roots[0].is_dir() and roots[1].is_dir()


def test_hermetic_fixtures_cannot_enter_production_official_run(
    tmp_path: Path,
) -> None:
    """Fixture checkpoint bytes must not satisfy production byte validation."""

    ckpt, fixture_digest = hermetic.write_hermetic_checkpoint(tmp_path / "ckpt.pth")
    assert fixture_digest != PRODUCTION_CHECKPOINT
    with pytest.raises(cache.TeacherCacheError, match="B2_CACHE_CHECKPOINT_HASH_MISMATCH"):
        cache.validate_checkpoint_bytes(ckpt, PRODUCTION_CHECKPOINT)


def test_production_checkpoint_and_split_hash_enforcement_unchanged() -> None:
    config = json.loads(
        (REPO_ROOT / "configs" / "phase_b" / "b2_teacher_cache_gate_c.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["checkpoint"]["sha256"] == PRODUCTION_CHECKPOINT
    assert config["split"]["scientific_sha256"] == PRODUCTION_SPLIT_V2
    fixture = hermetic.load_b2_split_fixture()
    assert (
        fixture["scientific_hash_contract"]["canonical_scientific_hash_v2"]
        == PRODUCTION_SPLIT_V2
    )


def test_release_tools_fail_closed_without_real_release_identities(
    tmp_path: Path,
) -> None:
    """Synthetic empty repos without release tags remain fail-closed."""

    repo = tmp_path / "empty-release"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-b", "main"], check=True)
    marker = repo / "README"
    marker.write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "README"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=ci",
            "-c",
            "user.email=ci@example.invalid",
            "commit",
            "-m",
            "init",
        ],
        check=True,
    )
    with pytest.raises(
        tiny_cli.B2TinySplitCLIError, match="B2_REPOSITORY_IDENTITY_UNAVAILABLE"
    ):
        tiny_cli._derive_repository_identity(
            repo,
            {
                "b1_base": {
                    "tag": EXPECTED_B1_TAG,
                    "commit": hermetic.EXPECTED_B1_COMMIT,
                }
            },
            require_clean=False,
        )
    with pytest.raises(
        teacher_cli.B2TeacherCacheCLIError,
        match="B2_CACHE_(REPOSITORY_IDENTITY_UNAVAILABLE|B2_TAG|CONTRACT_TAG)",
    ):
        teacher_cli._derive_repository_identity(repo, require_clean=False)


def test_descriptor_cli_suite_uses_hermetic_identity_repos_only() -> None:
    text = (REPO_ROOT / "tests" / "rad" / "test_b2_descriptor_artifacts_cli.py").read_text(
        encoding="utf-8"
    )
    assert "build_hermetic_descriptor_identity_repo" in text
    assert "_install_hermetic_identity" in text
    assert "b2-hermetic-main-integration-v1" in (
        REPO_ROOT / "tests" / "rad" / "b2_descriptor_fixtures.py"
    ).read_text(encoding="utf-8")


def test_descriptor_cli_hermetic_path_survives_when_real_checkout_tag_is_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CPU suite must succeed even when the real checkout cannot resolve the official tag."""

    from tests.rad import b2_descriptor_fixtures as desc_fixtures
    from tools import create_b2_descriptor_artifacts as desc_cli

    real_run_git = desc_cli._run_git
    official_tag = "b2-main-integration-v1"

    def hide_official_tag(repo: Path, *arguments: str):
        if (
            Path(repo).resolve() == REPO_ROOT.resolve()
            and arguments[:1] == ("rev-parse",)
            and any(official_tag in str(arg) for arg in arguments)
        ):
            return subprocess.CompletedProcess(
                args=["git", "-C", str(repo), *arguments],
                returncode=128,
                stdout="",
                stderr=(
                    f"fatal: ambiguous argument '{official_tag}^{{commit}}': "
                    "unknown revision or path not in the working tree.\n"
                ),
            )
        return real_run_git(repo, *arguments)

    monkeypatch.setattr(desc_cli, "_run_git", hide_official_tag)

    # Official production validation against the real checkout must fail closed.
    with pytest.raises(
        desc_cli.B2DescriptorArtifactsCLIError,
        match="B2_DESC_REPOSITORY_IDENTITY_UNAVAILABLE",
    ):
        desc_cli._validate_main_ancestry(
            REPO_ROOT,
            expected_main_tag=official_tag,
            expected_main_commit="51e18ade0231c7488ef582bde1e9694f933e85eb",
        )

    # Hermetic temporary repository path remains independently valid.
    identity = desc_fixtures.build_hermetic_descriptor_identity_repo(tmp_path)
    validated = desc_cli._validate_main_ancestry(
        identity["repo"],
        expected_main_tag=identity["expected_main_tag"],
        expected_main_commit=identity["expected_main_commit"],
    )
    assert validated["head_is_descendant"] is True
    assert validated["worktree_clean"] is True
    assert validated["head_commit"] == identity["head_commit"]


def test_cpu_suite_does_not_require_real_b2_main_integration_tag() -> None:
    probe = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "rev-parse",
            "b2-main-integration-v1^{commit}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode in {0, 128}

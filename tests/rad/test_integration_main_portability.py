"""Integration hygiene: portable configs before promote develop → main."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from rad.phase_b import b2_teacher_cache as cache

REPO_ROOT = Path(__file__).resolve().parents[2]
PRE_B105 = (
    REPO_ROOT
    / "docs"
    / "phase_b"
    / "b1_cuda_equivalence_manifest.pre_b105_candidate.json"
)
FINAL_B1 = REPO_ROOT / "docs" / "phase_b" / "b1_cuda_equivalence_manifest.json"
TEACHER_CACHE_CONFIG = REPO_ROOT / "configs" / "phase_b" / "b2_teacher_cache_gate_c.json"
TEACHER_CACHE_CLI = REPO_ROOT / "tools" / "create_b2_teacher_cache.py"
EXPECTED_CHECKPOINT_SHA256 = (
    "97bd461163efb96e36cddb1c3adf677e4c4fc2daabb2521021689f30e799b4f4"
)
EXPECTED_SPLIT_V2 = (
    "91570da1fed6d7859d407196b10403581832ae0ff677a1ea7657ca76b91471f0"
)
EXPECTED_PRE_B105_SHA256 = (
    "e09aa34c36718f95dd3d311bcc87192fb5948462aeb9c9318bbf6e7cfd793223"
)
GENERIC_RAD_CONFIGS = (
    REPO_ROOT / "configs" / "rad" / "adaptive.yaml",
    REPO_ROOT / "configs" / "rad" / "fusion.yaml",
    REPO_ROOT / "configs" / "rad" / "lse.yaml",
    REPO_ROOT / "configs" / "rad" / "experiments.yaml",
)


def _git_ls_files(path: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", str(path.relative_to(REPO_ROOT))],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return [line for line in proc.stdout.splitlines() if line.strip()]


def test_pre_b105_candidate_manifest_is_not_tracked() -> None:
    assert _git_ls_files(PRE_B105) == []
    assert not PRE_B105.exists()


def test_final_b1_manifest_documents_pre_b105_disposition_by_hash() -> None:
    payload = json.loads(FINAL_B1.read_text(encoding="utf-8"))
    invalidated = payload["invalidated_previous_evaluation"]["task_level_categories"]
    assert invalidated, "expected invalidated predecessor categories"
    evidence = invalidated[0]["retained_evidence"]
    assert isinstance(evidence, dict)
    assert evidence["sha256"] == EXPECTED_PRE_B105_SHA256
    assert evidence["disposition"] == "removed_superseded_candidate"
    assert "pre_b105_candidate.json" in evidence["former_path"]


def test_b2_teacher_cache_config_uses_portable_checkpoint_path() -> None:
    config = json.loads(TEACHER_CACHE_CONFIG.read_text(encoding="utf-8"))
    path = config["checkpoint"]["path"]
    assert isinstance(path, str)
    assert "/root/autodl-tmp/" not in path
    assert not path.startswith("/")
    assert config["checkpoint"]["sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert config["split"]["scientific_sha256"] == EXPECTED_SPLIT_V2
    canonical = hashlib.sha256(
        json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert canonical == cache._EXPECTED_CONFIG_CANONICAL_SHA256
    loaded = cache.load_teacher_cache_config(TEACHER_CACHE_CONFIG)
    assert loaded.checkpoint_sha256 == EXPECTED_CHECKPOINT_SHA256
    assert loaded.split_scientific_sha256 == EXPECTED_SPLIT_V2


def test_teacher_cache_cli_has_no_autodl_mvtec_default() -> None:
    text = TEACHER_CACHE_CLI.read_text(encoding="utf-8")
    assert "/root/autodl-tmp/" not in text
    assert "--mvtec-root" in text


def test_generic_rad_configs_avoid_machine_local_data_paths() -> None:
    for path in GENERIC_RAD_CONFIGS:
        text = path.read_text(encoding="utf-8")
        assert "/root/autodl-tmp/" not in text, path.name
        assert "data_path:" in text

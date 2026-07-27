"""Integration hygiene: portable configs before promote develop → main."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

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
FORBIDDEN_PORTABILITY_FRAGMENTS = (
    "/root/autodl-tmp/",
    "/root/miniconda3/",
    "AD-phase-",
)
ACTIVE_OPERATIONAL_RAD_CONFIGS = tuple(
    sorted((REPO_ROOT / "configs" / "rad").glob("*.yaml"))
)
CLI_DATA_PATH_CONTRACTS = (
    {
        "cli": REPO_ROOT / "tools" / "evaluate_adaptive_dataset.py",
        "default_config": REPO_ROOT / "configs" / "rad" / "adaptive.yaml",
        "explicit_args": ("--data-path",),
        "require_explicit": False,
    },
    {
        "cli": REPO_ROOT / "tools" / "reproduce_baseline.py",
        "default_config": REPO_ROOT / "configs" / "rad" / "baseline_mvtec_to_visa.yaml",
        "explicit_args": ("--train-data-path", "--test-data-path"),
        "require_explicit": True,
        "omit_fields": ("train.data_path", "test.data_path"),
    },
    {
        "cli": REPO_ROOT / "tools" / "evaluate_zero_shot_transfer.py",
        "default_config": REPO_ROOT / "configs" / "rad" / "zero_shot_transfer.yaml",
        "explicit_args": ("--target-data-path",),
        "require_explicit": True,
        "omit_fields": ("transfer.target_data_path",),
    },
    {
        "cli": REPO_ROOT / "tools" / "benchmark_latency.py",
        "default_config": REPO_ROOT / "configs" / "rad" / "benchmark.yaml",
        "explicit_args": (),
        "require_explicit": False,
    },
    {
        "cli": REPO_ROOT / "tools" / "create_b2_teacher_cache.py",
        "default_config": TEACHER_CACHE_CONFIG,
        "explicit_args": ("--mvtec-root",),
        "require_explicit": True,
    },
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


def _iter_operational_yaml_values(node: object, *, path: str = "") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            child_path = f"{path}.{key}" if path else str(key)
            found.extend(_iter_operational_yaml_values(value, path=child_path))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(
                _iter_operational_yaml_values(value, path=f"{path}[{index}]")
            )
    elif isinstance(node, str):
        found.append((path, node))
    return found


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


@pytest.mark.parametrize("config_path", ACTIVE_OPERATIONAL_RAD_CONFIGS)
def test_active_rad_configs_avoid_machine_local_operational_values(
    config_path: Path,
) -> None:
    import yaml

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for field_path, value in _iter_operational_yaml_values(payload):
        for fragment in FORBIDDEN_PORTABILITY_FRAGMENTS:
            assert fragment not in value, (
                f"{config_path.name}:{field_path} contains forbidden fragment "
                f"{fragment!r}: {value!r}"
            )


@pytest.mark.parametrize("contract", CLI_DATA_PATH_CONTRACTS, ids=lambda c: c["cli"].name)
def test_affected_clis_use_portable_defaults_or_require_explicit_paths(
    contract: dict[str, object],
) -> None:
    cli_path = contract["cli"]
    assert isinstance(cli_path, Path)
    text = cli_path.read_text(encoding="utf-8")
    for fragment in FORBIDDEN_PORTABILITY_FRAGMENTS:
        assert fragment not in text, f"{cli_path.name} contains {fragment!r}"

    explicit_args = contract["explicit_args"]
    assert isinstance(explicit_args, tuple)
    require_explicit = bool(contract["require_explicit"])
    if require_explicit:
        for arg in explicit_args:
            assert arg in text, f"{cli_path.name} must expose {arg}"

    default_config = contract["default_config"]
    assert isinstance(default_config, Path)
    if default_config.suffix == ".yaml":
        import yaml

        payload = yaml.safe_load(default_config.read_text(encoding="utf-8"))
        data_path_values = [
            value
            for field_path, value in _iter_operational_yaml_values(payload)
            if field_path.endswith("data_path")
        ]
        if require_explicit:
            omit_fields = tuple(contract.get("omit_fields", ()))
            for field_path, value in _iter_operational_yaml_values(payload):
                if field_path in omit_fields:
                    raise AssertionError(
                        f"{default_config.name} must omit {field_path} "
                        f"for CLI override (found {value!r})"
                    )
        else:
            for value in data_path_values:
                assert value.startswith("data/") or value.startswith("artifacts/"), (
                    f"{default_config.name} must use repo-relative data defaults: "
                    f"{value!r}"
                )

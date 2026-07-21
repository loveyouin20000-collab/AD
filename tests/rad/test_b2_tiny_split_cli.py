"""B2-01 RED: launcher-only tiny-split CLI and artifact publication contract.

The production CLI is never imported at collection time.  A missing CLI therefore
produces ordinary assertion failures, while controlled source records let the
tests exercise the production registry boundary without reading a real dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPO_ROOT / "tools" / "create_b2_tiny_split.py"
LAUNCHER_PATH = REPO_ROOT / "tools" / "run_with_execution_profile.py"
PROFILE_PATH = REPO_ROOT / "configs" / "execution" / "frozen_deterministic_math.json"
CONFIG_PATH = REPO_ROOT / "configs" / "phase_b" / "b2_tiny_gate_c.json"
ARTIFACT_BASE = REPO_ROOT / "artifacts" / "phase_b" / "b2_gate_c"
PRODUCTION_MVTEC_ROOT = Path("/root/autodl-tmp/data/mvtec")
EXPECTED_PROFILE_SHA256 = (
    "7af8dba39633743da0380fef9710940cded655f68c9efa8f84f5a52aeddb3c8d"
)
RESULT_PREFIX = "B2_TINY_SPLIT_RESULT="
AUDIT_PREFIX = "B2_TEST_AUDIT="
SUBPROCESS_TIMEOUT_SECONDS = 60
PASSED_STATUS = re.compile(r'"status"\s*:\s*"passed"', re.IGNORECASE)


def _test_run_id(tmp_path: Path) -> str:
    digest = hashlib.sha256(str(tmp_path).encode("utf-8")).hexdigest()[:12]
    return f"test-contract-{digest}"


@pytest.fixture(autouse=True)
def _cleanup_contract_run(tmp_path: Path) -> Any:
    yield
    shutil.rmtree(ARTIFACT_BASE / _test_run_id(tmp_path), ignore_errors=True)


def _require_cli() -> None:
    assert CLI_PATH.is_file(), f"B2-01 RED: missing production CLI: {CLI_PATH}"


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    for key in (
        "RAD_EXECUTION_PROFILE_BOOTSTRAPPED",
        "RAD_EXECUTION_PROFILE_PATH",
        "RAD_EXECUTION_PROFILE_SHA256",
    ):
        env.pop(key, None)
    return env


def _populate_controlled_source(root: Path) -> None:
    for category in ("bottle", "carpet"):
        for anomaly_type in ("good", "crack"):
            for index in range(8):
                image = root / category / "test" / anomaly_type / f"{index:03d}.png"
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(b"controlled-source-image")
                if anomaly_type != "good":
                    mask = (
                        root
                        / category
                        / "ground_truth"
                        / anomaly_type
                        / f"{index:03d}_mask.png"
                    )
                    mask.parent.mkdir(parents=True, exist_ok=True)
                    mask.write_bytes(b"controlled-source-mask")


def _cli_args(
    *,
    config: Path,
    source_root: Path,
    output_dir: Path,
    run_id: str = "b2-cli-contract",
    seed: int = 111,
    dry_run: bool = False,
) -> list[str]:
    args = [
        "--config",
        str(config),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--dataset-root",
        str(source_root),
        "--run-id",
        run_id,
        "--creation-timestamp",
        "2026-07-21T00:00:01Z",
    ]
    if dry_run:
        args.append("--dry-run")
    return args


def _harness_source(case: str) -> str:
    """Return a child harness loaded only after launcher bootstrap."""
    return f"""
import json
import os
import runpy
import sys
import tempfile
import builtins
from pathlib import Path

from rad.runtime.execution_profile import apply_execution_profile

attestation = apply_execution_profile()

from rad.data.adapters.types import EvaluationRecord
import rad.data.adapters.registry as registry
from rad.data.adapters.mvtec import MVTecAdapter

case = {case!r}
root = Path(sys.argv[1]).resolve()
cli_path = Path(sys.argv[2]).resolve()
cli_args = sys.argv[3:]
audit = {{"registry_calls": [], "open_image_calls": 0, "open_mask_calls": 0,
          "os_replace_calls": [], "output_mutations": [],
          "target_access_attempts": [],
          "attestation_provenance": None}}
declared_output = Path(
    cli_args[cli_args.index("--output-dir") + 1]
).resolve()

def inside_output(path):
    candidate = Path(path).resolve()
    return candidate == declared_output or declared_output in candidate.parents

def is_forbidden_target_path(path):
    if isinstance(path, int):
        return False
    return any(part.lower() == "visa" for part in Path(path).parts)

def record(category, label, index, *, mask=True, image_override=None, sample_suffix=""):
    anomaly_type = "good" if label == 0 else "crack"
    image = image_override or root / category / "test" / anomaly_type / f"{{index:03d}}.png"
    mask_path = None
    if label == 1 and mask:
        mask_path = root / category / "ground_truth" / anomaly_type / f"{{index:03d}}_mask.png"
    return EvaluationRecord(
        sample_id=image.relative_to(root).as_posix() + sample_suffix,
        dataset="mvtec",
        category=category,
        image_path=image,
        mask_path=mask_path,
        image_label=label,
        split="test",
    )

records = [
    record(category, label, index)
    for category in ("bottle", "carpet")
    for label in (0, 1)
    for index in range(8)
]
if case == "missing_mask":
    records[8] = record("bottle", 1, 0, mask=False)
elif case == "insufficient":
    records.pop(8)
elif case == "stable_collision":
    records[-1] = record(
        "bottle", 0, 0, image_override=records[0].image_path, sample_suffix="#duplicate"
    )

class ControlledMVTecAdapter(MVTecAdapter):
    def __init__(self):
        pass

    def records(self, split="test", *, categories=None):
        if split != "test":
            raise AssertionError("B2 tiny split may enumerate only the test source split")
        if categories is None:
            return tuple(records)
        selected = set(categories)
        return tuple(record for record in records if record.category in selected)

    def open_image(self, item):
        audit["open_image_calls"] += 1
        raise AssertionError("B2 tiny split must not open source images")

    def open_mask(self, item):
        audit["open_mask_calls"] += 1
        raise AssertionError("B2 tiny split must not open source masks")

def controlled_get_adapter(name, adapter_root):
    audit["registry_calls"].append([name, str(Path(adapter_root).resolve())])
    if name != "mvtec":
        raise AssertionError("target/VisA adapter access is forbidden")
    if Path(adapter_root).resolve() != root:
        raise AssertionError("registry received an unexpected dataset root")
    if case in ("valid", "atomic_failure"):
        return real_get_adapter(name, adapter_root)
    return ControlledMVTecAdapter()

def forbidden_visa(*args, **kwargs):
    raise AssertionError("VisA adapter construction is forbidden")

real_get_adapter = registry.get_adapter
registry.get_adapter = controlled_get_adapter
registry._ADAPTER_FACTORIES["visa"] = forbidden_visa

audit["attestation_provenance"] = dict(attestation.artifact_provenance())

real_replace = os.replace
def observed_replace(source, destination):
    audit["os_replace_calls"].append([str(source), str(destination)])
    if inside_output(source) or inside_output(destination):
        audit["output_mutations"].append(["replace", str(source), str(destination)])
    if case == "atomic_failure":
        raise OSError("injected atomic replace failure")
    return real_replace(source, destination)
os.replace = observed_replace

real_mkdir = Path.mkdir
def observed_mkdir(path, *args, **kwargs):
    if inside_output(path):
        audit["output_mutations"].append(["mkdir", str(path)])
    return real_mkdir(path, *args, **kwargs)
Path.mkdir = observed_mkdir

real_mkstemp = tempfile.mkstemp
def observed_mkstemp(*args, **kwargs):
    directory = kwargs.get("dir")
    if directory is not None and inside_output(directory):
        audit["output_mutations"].append(["mkstemp", str(directory)])
    return real_mkstemp(*args, **kwargs)
tempfile.mkstemp = observed_mkstemp

real_open = builtins.open
def observed_open(file, mode="r", *args, **kwargs):
    if isinstance(file, (str, bytes, os.PathLike)):
        if is_forbidden_target_path(file):
            audit["target_access_attempts"].append(["open", str(file)])
            raise AssertionError("VisA target path access is forbidden")
        if any(flag in mode for flag in "wax+") and inside_output(file):
            audit["output_mutations"].append(["open", str(file), mode])
    return real_open(file, mode, *args, **kwargs)
builtins.open = observed_open

real_iterdir = Path.iterdir
def observed_iterdir(path):
    if is_forbidden_target_path(path):
        audit["target_access_attempts"].append(["iterdir", str(path)])
        raise AssertionError("VisA target enumeration is forbidden")
    return real_iterdir(path)
Path.iterdir = observed_iterdir

real_scandir = os.scandir
def observed_scandir(path):
    if is_forbidden_target_path(path):
        audit["target_access_attempts"].append(["scandir", str(path)])
        raise AssertionError("VisA target enumeration is forbidden")
    return real_scandir(path)
os.scandir = observed_scandir

exit_code = 0
try:
    module = runpy.run_path(str(cli_path), run_name="b2_cli_contract")
    module["main"].__globals__["_apply_profile"] = lambda _repo: attestation
    exit_code = module["main"](cli_args)
finally:
    print({AUDIT_PREFIX!r} + json.dumps(audit, sort_keys=True))
raise SystemExit(exit_code)
"""


def _run_cli(
    tmp_path: Path,
    *,
    case: str = "valid",
    config: Path = CONFIG_PATH,
    source_root: Path | None = None,
    output_dir: Path | None = None,
    run_id: str | None = None,
    seed: int = 111,
    dry_run: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    _require_cli()
    root = source_root or PRODUCTION_MVTEC_ROOT
    out = output_dir or ARTIFACT_BASE
    effective_run_id = run_id or _test_run_id(tmp_path)
    if root.resolve() != PRODUCTION_MVTEC_ROOT.resolve():
        _populate_controlled_source(root)
    child_args = _cli_args(
        config=config,
        source_root=root,
        output_dir=out,
        run_id=effective_run_id,
        seed=seed,
        dry_run=dry_run,
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER_PATH),
            "--profile",
            str(PROFILE_PATH),
            "--expected-sha256",
            EXPECTED_PROFILE_SHA256,
            "--",
            sys.executable,
            "-c",
            _harness_source(case),
            str(root),
            str(CLI_PATH),
            *child_args,
        ],
        cwd=REPO_ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    return proc, root, out


def _prefixed_json(proc: subprocess.CompletedProcess[str], prefix: str) -> dict[str, Any]:
    matches = [
        json.loads(line.removeprefix(prefix))
        for line in proc.stdout.splitlines()
        if line.startswith(prefix)
    ]
    assert len(matches) == 1, (
        f"expected one {prefix!r} record, got {len(matches)}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    assert isinstance(matches[0], dict)
    return matches[0]


def _result(proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return _prefixed_json(proc, RESULT_PREFIX)


def _audit(proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return _prefixed_json(proc, AUDIT_PREFIX)


def _all_paths(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {path.relative_to(root).as_posix() for path in root.rglob("*")}


def _assert_no_passed_manifest(
    output_dir: Path,
    run_id: str,
    proc: subprocess.CompletedProcess[str],
) -> None:
    assert proc.returncode != 0, proc.stdout + proc.stderr
    run_dir = output_dir / run_id
    for path in run_dir.rglob("*.json") if run_dir.exists() else ():
        assert not PASSED_STATUS.search(path.read_text(encoding="utf-8")), path


def _write_changed_config(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None]
) -> Path:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(config)
    path = tmp_path / "changed-b2-config.json"
    path.write_text(
        json.dumps(config, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


def test_cli_help_exposes_required_inputs_and_no_forgeable_attestation_fields() -> None:
    _require_cli()
    proc = subprocess.run(
        [sys.executable, str(CLI_PATH), "--help"],
        cwd=REPO_ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for option in (
        "--config",
        "--seed",
        "--output-dir",
        "--dry-run",
        "--dataset-root",
        "--run-id",
        "--creation-timestamp",
    ):
        assert option in proc.stdout
    for forbidden in (
        "--execution-profile-id",
        "--execution-profile-sha256",
        "--runtime-attestation-sha256",
    ):
        assert forbidden not in proc.stdout


def test_direct_invocation_fails_before_registry_or_artifact_creation(tmp_path: Path) -> None:
    _require_cli()
    root = tmp_path / "controlled_mvtec"
    output = ARTIFACT_BASE
    run_id = _test_run_id(tmp_path)
    _populate_controlled_source(root)
    proc = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            *_cli_args(
                config=CONFIG_PATH,
                source_root=root,
                output_dir=output,
                run_id=run_id,
                dry_run=True,
            ),
        ],
        cwd=REPO_ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    assert proc.returncode != 0
    assert "B2_BOOTSTRAP_MARKER_MISSING" in proc.stdout + proc.stderr
    assert not (output / run_id).exists()


def test_valid_dry_run_performs_complete_in_memory_validation_and_writes_nothing(
    tmp_path: Path,
) -> None:
    proc, root, output = _run_cli(tmp_path, dry_run=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = _result(proc)
    audit = _audit(proc)
    assert payload["mode"] == "dry-run"
    assert payload["canonical_scientific_sha256"] == hashlib.sha256(
        json.dumps(
            payload["canonical_scientific_content"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert payload["official_manifest"]["source"]["dataset"] == "mvtec"
    assert payload["official_manifest"]["status"] == "passed"
    validation = payload["validation"]
    assert set(validation) == {
        "execution_profile_attestation",
        "production_adapter_enumeration",
        "selection",
        "audits",
    }
    assert set(validation["execution_profile_attestation"]) == {
        "schema_version",
        "profile",
        "requested_settings",
        "effective_settings",
        "environment",
        "canary",
    }
    assert validation["execution_profile_attestation"]["profile"]["hashes_match"] is True
    enumeration = validation["production_adapter_enumeration"]
    assert enumeration["dataset"] == "mvtec"
    assert enumeration["root"] == str(root.resolve())
    assert enumeration["split"] == "test"
    assert enumeration["adapter_module"] == "rad.data.adapters.mvtec"
    assert enumeration["adapter_class"] == "MVTecAdapter"
    assert enumeration["record_count"] > 32
    assert validation["selection"]["split_counts"] == {
        "training": 16,
        "calibration": 8,
        "evaluation": 8,
    }
    assert validation["selection"]["selected_count"] == 32
    assert validation["audits"] and all(validation["audits"].values())
    assert audit["registry_calls"] == [["mvtec", str(root.resolve())]]
    assert audit["open_image_calls"] == audit["open_mask_calls"] == 0
    assert audit["target_access_attempts"] == []
    assert audit["os_replace_calls"] == []
    assert audit["output_mutations"] == []
    assert not (output / _test_run_id(tmp_path)).exists()
    assert not any(
        name.endswith((".tmp", ".temp", ".lock")) or "manifest" in name
        for name in _all_paths(ARTIFACT_BASE / _test_run_id(tmp_path))
    )


def test_dry_run_hash_equals_subsequent_official_hash_and_manifest_is_unique(
    tmp_path: Path,
) -> None:
    dry_proc, root, output = _run_cli(tmp_path, dry_run=True)
    assert dry_proc.returncode == 0, dry_proc.stdout + dry_proc.stderr
    dry_payload = _result(dry_proc)
    official_proc, _, _ = _run_cli(
        tmp_path,
        source_root=root,
        output_dir=output,
    )
    assert official_proc.returncode == 0, official_proc.stdout + official_proc.stderr
    official_payload = _result(official_proc)
    run_id = _test_run_id(tmp_path)
    manifests = list((output / run_id).rglob("split_manifest.json"))
    assert manifests == [output / run_id / "split_manifest.json"]
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest == official_payload["official_manifest"]
    assert dry_payload["canonical_scientific_sha256"] == (
        official_payload["canonical_scientific_sha256"]
    )
    attested = _audit(official_proc)["attestation_provenance"]
    assert set(attested) == {
        "execution_profile_name",
        "execution_profile_sha256",
        "runtime_attestation_sha256",
    }
    assert manifest["execution_profile"] == attested
    assert manifest["output_directory"] == str((output / run_id).resolve())


def test_repeated_dry_run_has_no_collision_lock_temp_or_atomic_replace(
    tmp_path: Path,
) -> None:
    first, root, output = _run_cli(tmp_path, dry_run=True)
    second, _, _ = _run_cli(
        tmp_path,
        source_root=root,
        output_dir=output,
        dry_run=True,
    )
    assert first.returncode == second.returncode == 0
    assert _result(first)["canonical_scientific_sha256"] == (
        _result(second)["canonical_scientific_sha256"]
    )
    assert _audit(first)["os_replace_calls"] == _audit(second)["os_replace_calls"] == []
    assert _audit(first)["output_mutations"] == _audit(second)["output_mutations"] == []
    assert not (output / _test_run_id(tmp_path)).exists()


def test_official_write_is_atomic_and_existing_run_directory_fails_closed(
    tmp_path: Path,
) -> None:
    first, root, output = _run_cli(tmp_path)
    assert first.returncode == 0, first.stdout + first.stderr
    replace_calls = _audit(first)["os_replace_calls"]
    assert len(replace_calls) == 1
    source, destination = map(Path, replace_calls[0])
    assert destination == output / _test_run_id(tmp_path) / "split_manifest.json"
    assert source.parent == destination.parent
    assert source != destination
    assert not source.exists()
    before = destination.read_bytes()

    collision, _, _ = _run_cli(tmp_path, source_root=root, output_dir=output)
    assert collision.returncode != 0
    assert "B2_OUTPUT_COLLISION" in collision.stdout + collision.stderr
    assert _audit(collision)["os_replace_calls"] == []
    assert destination.read_bytes() == before
    assert not any(
        path.name.endswith((".tmp", ".temp", ".lock"))
        for path in output.rglob("*")
    )


def test_preexisting_empty_run_directory_fails_before_source_enumeration(
    tmp_path: Path,
) -> None:
    run_dir = ARTIFACT_BASE / _test_run_id(tmp_path)
    run_dir.mkdir(parents=True)
    proc, _, _ = _run_cli(tmp_path)
    assert proc.returncode != 0
    assert "B2_OUTPUT_COLLISION" in proc.stdout + proc.stderr
    assert _audit(proc)["registry_calls"] == []
    assert _audit(proc)["os_replace_calls"] == []
    assert list(run_dir.iterdir()) == []


def test_atomic_replace_failure_leaves_no_manifest_or_temporary_file(
    tmp_path: Path,
) -> None:
    proc, _, output = _run_cli(tmp_path, case="atomic_failure")
    run_dir = output / _test_run_id(tmp_path)
    assert proc.returncode != 0
    assert "B2_ATOMIC_WRITE_FAILED" in proc.stdout + proc.stderr
    assert not (run_dir / "split_manifest.json").exists()
    assert not any(
        path.name.endswith((".tmp", ".temp", ".lock"))
        for path in run_dir.rglob("*")
    )


@pytest.mark.parametrize(
    ("case", "root_component", "expected_code"),
    [
        ("missing_mask", "controlled_mvtec", "B2_ANOMALOUS_MASK_MISSING"),
        ("insufficient", "controlled_mvtec", "B2_INSUFFICIENT_STRATUM"),
        ("stable_collision", "controlled_mvtec", "B2_STABLE_ID_COLLISION"),
        ("valid", "fixtures", "B2_FORBIDDEN_SOURCE_PATH"),
        ("valid", "examples", "B2_FORBIDDEN_SOURCE_PATH"),
    ],
)
def test_invalid_source_state_is_nonzero_without_passed_manifest(
    case: str,
    root_component: str,
    expected_code: str,
    tmp_path: Path,
) -> None:
    output = ARTIFACT_BASE
    run_id = _test_run_id(tmp_path)
    proc, root, _ = _run_cli(
        tmp_path,
        case=case,
        source_root=tmp_path / root_component / "mvtec",
        output_dir=output,
    )
    assert root.exists()
    assert proc.returncode == 1
    _assert_no_passed_manifest(output, run_id, proc)
    assert expected_code in proc.stdout + proc.stderr
    audit = _audit(proc)
    if root_component in {"fixtures", "examples"}:
        assert audit["registry_calls"] == []
    else:
        assert audit["registry_calls"] == [["mvtec", str(root.resolve())]]
    assert audit["os_replace_calls"] == []


def test_target_dataset_config_is_rejected_without_enumerating_visa(
    tmp_path: Path,
) -> None:
    config = _write_changed_config(
        tmp_path,
        lambda data: data.update(
            {"source_dataset": "visa", "transfer_direction": "visa_to_mvtec"}
        ),
    )
    output = ARTIFACT_BASE
    run_id = _test_run_id(tmp_path)
    proc, _, _ = _run_cli(tmp_path, config=config, output_dir=output)
    _assert_no_passed_manifest(output, run_id, proc)
    assert "B2_TARGET_DATASET_FORBIDDEN" in proc.stdout + proc.stderr
    audit = _audit(proc)
    assert all(call[0] != "visa" for call in audit["registry_calls"])
    assert audit["target_access_attempts"] == []
    assert audit["os_replace_calls"] == []


def test_visa_category_in_config_is_rejected_before_source_enumeration(
    tmp_path: Path,
) -> None:
    config = _write_changed_config(
        tmp_path,
        lambda data: data["categories"].append("candle"),
    )
    output = ARTIFACT_BASE
    run_id = _test_run_id(tmp_path)
    proc, _, _ = _run_cli(tmp_path, config=config, output_dir=output)
    _assert_no_passed_manifest(output, run_id, proc)
    assert "B2_TARGET_CATEGORY_FORBIDDEN" in proc.stdout + proc.stderr
    audit = _audit(proc)
    assert audit["registry_calls"] == []
    assert audit["target_access_attempts"] == []
    assert audit["os_replace_calls"] == []


def test_changed_profile_hash_is_rejected_without_official_write(
    tmp_path: Path,
) -> None:
    config = _write_changed_config(
        tmp_path,
        lambda data: data["execution_profile"].update({"sha256": "0" * 64}),
    )
    output = ARTIFACT_BASE
    run_id = _test_run_id(tmp_path)
    proc, _, _ = _run_cli(tmp_path, config=config, output_dir=output)
    _assert_no_passed_manifest(output, run_id, proc)
    assert "B2_EXECUTION_PROFILE_MISMATCH" in proc.stdout + proc.stderr
    assert _audit(proc)["os_replace_calls"] == []


def test_seed_drift_is_rejected_without_official_write(tmp_path: Path) -> None:
    output = ARTIFACT_BASE
    run_id = _test_run_id(tmp_path)
    proc, _, _ = _run_cli(tmp_path, seed=112, output_dir=output)
    _assert_no_passed_manifest(output, run_id, proc)
    assert "B2_SEED_DRIFT" in proc.stdout + proc.stderr
    assert _audit(proc)["os_replace_calls"] == []


def test_official_output_outside_b2_artifact_tree_is_rejected(
    tmp_path: Path,
) -> None:
    invalid_output = tmp_path / "outside-b2-artifacts"
    proc, _, _ = _run_cli(tmp_path, output_dir=invalid_output)
    assert proc.returncode != 0
    assert "B2_OUTPUT_LOCATION_INVALID" in proc.stdout + proc.stderr
    assert _audit(proc)["registry_calls"] == []
    assert _audit(proc)["os_replace_calls"] == []
    assert not invalid_output.exists()

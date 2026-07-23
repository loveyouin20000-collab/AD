"""B2-00 RED: stdlib-only pre-import execution-profile launcher contract."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = REPO_ROOT / "tools" / "run_with_execution_profile.py"
PROFILE_PATH = REPO_ROOT / "configs" / "execution" / "frozen_deterministic_math.json"
EXPECTED_PROFILE_SHA256 = (
    "7af8dba39633743da0380fef9710940cded655f68c9efa8f84f5a52aeddb3c8d"
)
BOOTSTRAP_MARKER = "RAD_EXECUTION_PROFILE_BOOTSTRAPPED"
PROFILE_PATH_ENV = "RAD_EXECUTION_PROFILE_PATH"
PROFILE_SHA_ENV = "RAD_EXECUTION_PROFILE_SHA256"
FORBIDDEN_IMPORT_PREFIXES = ("torch", "torchvision", "clip", "open_clip", "rad", "VisualAD_lib")
SUBPROCESS_TIMEOUT_SECONDS = 30
ERROR_RECORD = re.compile(r"^B2_PROFILE_ERROR\[([A-Z][A-Z0-9_]*)\]: .+$")
FROZEN_PROFILE = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    for key in (BOOTSTRAP_MARKER, PROFILE_PATH_ENV, PROFILE_SHA_ENV):
        env.pop(key, None)
    return env


def _run_launcher(
    *child_argv: str,
    profile: Path = PROFILE_PATH,
    expected_sha256: str = EXPECTED_PROFILE_SHA256,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    assert LAUNCHER_PATH.is_file(), f"missing production launcher: {LAUNCHER_PATH}"
    return subprocess.run(
        [
            sys.executable,
            str(LAUNCHER_PATH),
            "--profile",
            str(profile),
            "--expected-sha256",
            expected_sha256,
            "--",
            *child_argv,
        ],
        cwd=REPO_ROOT,
        env=_clean_env() if env is None else env,
        capture_output=True,
        text=True,
        check=False,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )


def _structured_error_codes(proc: subprocess.CompletedProcess[str]) -> list[str]:
    return [
        match.group(1)
        for line in (*proc.stdout.splitlines(), *proc.stderr.splitlines())
        if (match := ERROR_RECORD.fullmatch(line))
    ]


def _assert_cli_failure(
    proc: subprocess.CompletedProcess[str], expected_codes: str | set[str]
) -> None:
    assert proc.returncode != 0
    expected = {expected_codes} if isinstance(expected_codes, str) else expected_codes
    observed = set(_structured_error_codes(proc))
    assert observed == expected, (
        f"expected exact structured error codes {sorted(expected)!r}; "
        f"observed codes={sorted(observed)!r}\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )


def test_structured_error_parser_uses_complete_records_and_accepts_multiple_codes() -> None:
    proc = subprocess.CompletedProcess(
        args=[],
        returncode=1,
        stdout=(
            "prefix B2_PROFILE_ERROR[B2_FALSE_POSITIVE]: not a complete record\n"
            "B2_PROFILE_ERROR[B2_FIRST]: first detail\n"
        ),
        stderr="B2_PROFILE_ERROR[B2_SECOND]: second detail\n",
    )
    assert _structured_error_codes(proc) == ["B2_FIRST", "B2_SECOND"]


def _write_profile(tmp_path: Path, transform: Any) -> Path:
    data = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    transform(data)
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(data, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


def test_launcher_source_has_only_standard_library_imports() -> None:
    assert LAUNCHER_PATH.is_file()
    tree = ast.parse(LAUNCHER_PATH.read_text(encoding="utf-8"), filename=str(LAUNCHER_PATH))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    assert imported_roots <= stdlib
    assert not imported_roots.intersection(
        prefix.split(".", 1)[0] for prefix in FORBIDDEN_IMPORT_PREFIXES
    )
    dynamic_import_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in {
            "__import__",
            "import_module",
            "exec",
            "eval",
        }:
            dynamic_import_calls.append(node)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"__import__", "import_module"}
        ):
            dynamic_import_calls.append(node)
    assert not dynamic_import_calls, "launcher must not use dynamic import calls"


def test_importing_launcher_under_sentinels_has_no_forbidden_transitive_imports() -> None:
    assert LAUNCHER_PATH.is_file()
    probe = f"""
import builtins
import importlib.util
import json
import os
import sys
import sysconfig

path = {str(LAUNCHER_PATH)!r}
forbidden = {FORBIDDEN_IMPORT_PREFIXES!r}
attempts = []
real_import = builtins.__import__
baseline_modules = set(sys.modules)
stdlib_roots = [
    os.path.realpath(sysconfig.get_path("stdlib")),
    os.path.realpath(sysconfig.get_path("platstdlib")),
]
site_roots = [
    os.path.realpath(sysconfig.get_path("purelib")),
    os.path.realpath(sysconfig.get_path("platlib")),
]

def is_under(candidate, root):
    try:
        return os.path.commonpath([candidate, root]) == root
    except ValueError:
        return False

def assert_new_modules_are_stdlib():
    violations = []
    for module_name in sorted(set(sys.modules) - baseline_modules):
        if module_name == "b2_launcher_import_probe":
            continue
        module = sys.modules[module_name]
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        if origin in (None, "built-in", "frozen"):
            continue
        resolved = os.path.realpath(origin)
        in_stdlib = any(is_under(resolved, root) for root in stdlib_roots)
        in_site = any(is_under(resolved, root) for root in site_roots)
        if not in_stdlib or in_site:
            violations.append({{"module": module_name, "origin": resolved}})
    if violations:
        raise AssertionError("non-stdlib launcher imports: " + json.dumps(violations))

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if any(name == item or name.startswith(item + ".") for item in forbidden):
        attempts.append(name)
        raise AssertionError("forbidden pre-launch import: " + name)
    imported = real_import(name, globals, locals, fromlist, level)
    assert_new_modules_are_stdlib()
    return imported

builtins.__import__ = guarded_import
spec = importlib.util.spec_from_file_location("b2_launcher_import_probe", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert_new_modules_are_stdlib()
loaded = sorted(
    name for name in sys.modules
    if any(name == item or name.startswith(item + ".") for item in forbidden)
)
print(json.dumps({{"attempts": attempts, "loaded": loaded}}))
"""
    proc = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        cwd=REPO_ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert json.loads(lines[-1]) == {"attempts": [], "loaded": []}


def test_legal_launcher_process_loads_only_stdlib_before_bootstrap(
    tmp_path: Path,
) -> None:
    audit_log = tmp_path / "launcher-imports.json"
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        """
import atexit
import builtins
import json
import os
import sys
import sysconfig

if os.environ.get("RAD_EXECUTION_PROFILE_BOOTSTRAPPED") != "1":
    audit_path = os.environ["B2_IMPORT_AUDIT_LOG"]
    baseline = set(sys.modules)
    violations = {}
    real_import = builtins.__import__
    stdlib_roots = {
        os.path.realpath(sysconfig.get_path("stdlib")),
        os.path.realpath(sysconfig.get_path("platstdlib")),
    }
    site_roots = {
        os.path.realpath(sysconfig.get_path("purelib")),
        os.path.realpath(sysconfig.get_path("platlib")),
    }

    def is_under(candidate, root):
        try:
            return os.path.commonpath([candidate, root]) == root
        except ValueError:
            return False

    def audit_loaded_modules():
        for module_name in sorted(set(sys.modules) - baseline):
            module = sys.modules.get(module_name)
            origin = getattr(getattr(module, "__spec__", None), "origin", None)
            if origin in (None, "built-in", "frozen"):
                continue
            resolved = os.path.realpath(origin)
            in_stdlib = any(is_under(resolved, root) for root in stdlib_roots)
            in_site = any(is_under(resolved, root) for root in site_roots)
            if not in_stdlib or in_site:
                violations[module_name] = resolved

    def audited_import(name, globals=None, locals=None, fromlist=(), level=0):
        imported = real_import(name, globals, locals, fromlist, level)
        audit_loaded_modules()
        return imported

    def write_audit():
        audit_loaded_modules()
        with open(audit_path, "w", encoding="utf-8") as handle:
            json.dump(violations, handle, sort_keys=True)

    builtins.__import__ = audited_import
    atexit.register(write_audit)
""",
        encoding="utf-8",
    )
    env = _clean_env()
    env["PYTHONPATH"] = str(tmp_path)
    env["B2_IMPORT_AUDIT_LOG"] = str(audit_log)
    proc = _run_launcher(
        sys.executable,
        "-c",
        "raise SystemExit(0)",
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert audit_log.is_file()
    assert json.loads(audit_log.read_text(encoding="utf-8")) == {}


def test_launcher_accepts_exact_profile_bytes_and_establishes_environment_before_child() -> None:
    child = (
        "import hashlib,json,os,sys; "
        "p=os.environ['RAD_EXECUTION_PROFILE_PATH']; "
        "print(json.dumps({"
        "'marker':os.environ['RAD_EXECUTION_PROFILE_BOOTSTRAPPED'],"
        "'path':p,"
        "'exported_hash':os.environ['RAD_EXECUTION_PROFILE_SHA256'],"
        "'current_hash':hashlib.sha256(open(p,'rb').read()).hexdigest(),"
        "'cublas':os.environ['CUBLAS_WORKSPACE_CONFIG'],"
        "'forbidden_loaded':[x for x in sys.modules if "
        "x.split('.')[0] in {'torch','torchvision','clip','rad','VisualAD_lib'}]"
        "},sort_keys=True))"
    )
    proc = _run_launcher(sys.executable, "-I", "-c", child)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads([line for line in proc.stdout.splitlines() if line.strip()][-1])
    assert payload == {
        "marker": "1",
        "path": str(PROFILE_PATH.resolve()),
        "exported_hash": EXPECTED_PROFILE_SHA256,
        "current_hash": EXPECTED_PROFILE_SHA256,
        "cublas": ":4096:8",
        "forbidden_loaded": [],
    }


def test_launcher_preserves_child_argv_without_shell_reinterpretation() -> None:
    values = ["argument with spaces", "$(must-not-expand)", "semi;colon", "*"]
    proc = _run_launcher(
        sys.executable,
        "-c",
        "import json,sys; print(json.dumps(sys.argv[1:]))",
        *values,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads([line for line in proc.stdout.splitlines() if line.strip()][-1]) == values


@pytest.mark.parametrize("exit_code", [0, 3, 17])
def test_launcher_propagates_child_exit_code_exactly(exit_code: int) -> None:
    proc = _run_launcher(
        sys.executable,
        "-c",
        f"raise SystemExit({exit_code})",
    )
    assert proc.returncode == exit_code


def test_launcher_rejects_missing_profile_file(tmp_path: Path) -> None:
    proc = _run_launcher(
        sys.executable,
        "-c",
        "raise SystemExit(0)",
        profile=tmp_path / "missing.json",
    )
    _assert_cli_failure(proc, "B2_PROFILE_FILE_MISSING")


def test_launcher_rejects_incorrect_expected_hash() -> None:
    proc = _run_launcher(
        sys.executable,
        "-c",
        "raise SystemExit(0)",
        expected_sha256="0" * 64,
    )
    _assert_cli_failure(proc, "B2_PROFILE_HASH_MISMATCH")


def test_launcher_rejects_reformatted_profile_even_with_matching_supplied_hash(
    tmp_path: Path,
) -> None:
    reformatted = tmp_path / "reformatted-approved-profile.json"
    reformatted.write_text(
        json.dumps(FROZEN_PROFILE, indent=4, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reformatted_sha256 = hashlib.sha256(reformatted.read_bytes()).hexdigest()
    assert reformatted_sha256 != EXPECTED_PROFILE_SHA256

    proc = _run_launcher(
        sys.executable,
        "-c",
        "raise SystemExit(0)",
        profile=reformatted,
        expected_sha256=reformatted_sha256,
    )
    _assert_cli_failure(proc, "B2_PROFILE_IDENTITY_NOT_APPROVED")


def test_launcher_rejects_malformed_json(tmp_path: Path) -> None:
    profile = tmp_path / "malformed.json"
    profile.write_text('{"schema_version":', encoding="utf-8")
    digest = hashlib.sha256(profile.read_bytes()).hexdigest()
    proc = _run_launcher(
        sys.executable,
        "-c",
        "raise SystemExit(0)",
        profile=profile,
        expected_sha256=digest,
    )
    _assert_cli_failure(proc, "B2_PROFILE_JSON_INVALID")


@pytest.mark.parametrize(
    ("node", "key"),
    [
        *(("top", key) for key in FROZEN_PROFILE),
        *(
            (node, key)
            for node in ("required_environment", "torch", "model")
            for key in FROZEN_PROFILE[node]
        ),
    ],
)
def test_launcher_rejects_deletion_of_every_frozen_schema_key(
    tmp_path: Path,
    node: str,
    key: str,
) -> None:
    def delete_key(data: dict[str, Any]) -> None:
        if node == "top":
            data.pop(key)
        else:
            data[node].pop(key)

    profile = _write_profile(tmp_path, delete_key)
    proc = _run_launcher(
        sys.executable,
        "-c",
        "raise SystemExit(0)",
        profile=profile,
        expected_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
    )
    _assert_cli_failure(proc, "B2_PROFILE_SCHEMA_MISSING")


@pytest.mark.parametrize("field", ["applies_to", "notes"])
@pytest.mark.parametrize("mutation", ["content", "type"])
def test_launcher_rejects_drift_of_every_applies_to_and_notes_element(
    tmp_path: Path,
    field: str,
    mutation: str,
) -> None:
    for index in range(len(FROZEN_PROFILE[field])):
        def mutate_element(data: dict[str, Any], *, item_index: int = index) -> None:
            data[field][item_index] = (
                f"changed-{item_index}" if mutation == "content" else item_index
            )

        profile = _write_profile(tmp_path, mutate_element)
        proc = _run_launcher(
            sys.executable,
            "-c",
            "raise SystemExit(0)",
            profile=profile,
            expected_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
        )
        expected_code = (
            "B2_PROFILE_VALUE_UNSUPPORTED"
            if mutation == "content"
            else "B2_PROFILE_SCHEMA_TYPE"
        )
        _assert_cli_failure(proc, expected_code)


@pytest.mark.parametrize("field", ["applies_to", "notes"])
@pytest.mark.parametrize("operation", ["shorter", "longer"])
def test_launcher_rejects_applies_to_and_notes_length_drift(
    tmp_path: Path,
    field: str,
    operation: str,
) -> None:
    def change_length(data: dict[str, Any]) -> None:
        if operation == "shorter":
            data[field].pop()
        else:
            data[field].append(data[field][-1])

    profile = _write_profile(tmp_path, change_length)
    proc = _run_launcher(
        sys.executable,
        "-c",
        "raise SystemExit(0)",
        profile=profile,
        expected_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
    )
    _assert_cli_failure(proc, "B2_PROFILE_VALUE_UNSUPPORTED")


@pytest.mark.parametrize(
    ("case", "transform", "requirement_code"),
    [
        (
            "top-level missing",
            lambda data: data.pop("schema_version"),
            "B2_PROFILE_SCHEMA_MISSING",
        ),
        (
            "top-level unknown",
            lambda data: data.update({"untracked_top_level": True}),
            "B2_PROFILE_SCHEMA_UNKNOWN",
        ),
        (
            "top-level wrong type",
            lambda data: data.update({"schema_version": "1"}),
            "B2_PROFILE_SCHEMA_TYPE",
        ),
        (
            "schema version unsupported",
            lambda data: data.update({"schema_version": 2}),
            "B2_PROFILE_VALUE_UNSUPPORTED",
        ),
        (
            "profile_id missing",
            lambda data: data.pop("profile_id"),
            "B2_PROFILE_SCHEMA_MISSING",
        ),
        (
            "profile_id wrong type",
            lambda data: data.update({"profile_id": 7}),
            "B2_PROFILE_SCHEMA_TYPE",
        ),
        (
            "profile_id content drift",
            lambda data: data.update({"profile_id": "other_profile"}),
            "B2_PROFILE_VALUE_UNSUPPORTED",
        ),
        (
            "required_environment node missing",
            lambda data: data.pop("required_environment"),
            "B2_PROFILE_SCHEMA_MISSING",
        ),
        (
            "required_environment missing",
            lambda data: data["required_environment"].pop("CUBLAS_WORKSPACE_CONFIG"),
            "B2_PROFILE_SCHEMA_MISSING",
        ),
        (
            "required_environment unknown",
            lambda data: data["required_environment"].update({"UNTRACKED_ENV": "1"}),
            "B2_PROFILE_SCHEMA_UNKNOWN",
        ),
        (
            "required_environment wrong type",
            lambda data: data.update({"required_environment": []}),
            "B2_PROFILE_SCHEMA_TYPE",
        ),
        (
            "required_environment value wrong type",
            lambda data: data["required_environment"].update(
                {"CUBLAS_WORKSPACE_CONFIG": 4096}
            ),
            "B2_PROFILE_SCHEMA_TYPE",
        ),
        (
            "torch node missing",
            lambda data: data.pop("torch"),
            "B2_PROFILE_SCHEMA_MISSING",
        ),
        (
            "torch node wrong type",
            lambda data: data.update({"torch": []}),
            "B2_PROFILE_SCHEMA_TYPE",
        ),
        (
            "torch missing",
            lambda data: data["torch"].pop("seed"),
            "B2_PROFILE_SCHEMA_MISSING",
        ),
        (
            "torch unknown",
            lambda data: data["torch"].update({"untracked_backend": False}),
            "B2_PROFILE_SCHEMA_UNKNOWN",
        ),
        (
            "torch wrong type",
            lambda data: data["torch"].update({"seed": "111"}),
            "B2_PROFILE_SCHEMA_TYPE",
        ),
        (
            "model node missing",
            lambda data: data.pop("model"),
            "B2_PROFILE_SCHEMA_MISSING",
        ),
        (
            "model node wrong type",
            lambda data: data.update({"model": []}),
            "B2_PROFILE_SCHEMA_TYPE",
        ),
        (
            "model missing",
            lambda data: data["model"].pop("dtype"),
            "B2_PROFILE_SCHEMA_MISSING",
        ),
        (
            "model unknown",
            lambda data: data["model"].update({"untracked_model_flag": True}),
            "B2_PROFILE_SCHEMA_UNKNOWN",
        ),
        (
            "model wrong type",
            lambda data: data["model"].update({"amp": "false"}),
            "B2_PROFILE_SCHEMA_TYPE",
        ),
        (
            "applies_to missing",
            lambda data: data.pop("applies_to"),
            "B2_PROFILE_SCHEMA_MISSING",
        ),
        (
            "applies_to content drift",
            lambda data: data["applies_to"].__setitem__(0, "B0"),
            "B2_PROFILE_VALUE_UNSUPPORTED",
        ),
        (
            "applies_to element wrong type",
            lambda data: data["applies_to"].__setitem__(0, 3),
            "B2_PROFILE_SCHEMA_TYPE",
        ),
        (
            "applies_to wrong type",
            lambda data: data.update({"applies_to": "B2"}),
            "B2_PROFILE_SCHEMA_TYPE",
        ),
        (
            "notes missing",
            lambda data: data.pop("notes"),
            "B2_PROFILE_SCHEMA_MISSING",
        ),
        (
            "notes content drift",
            lambda data: data["notes"].__setitem__(0, "changed note"),
            "B2_PROFILE_VALUE_UNSUPPORTED",
        ),
        (
            "notes element wrong type",
            lambda data: data["notes"].__setitem__(0, 3),
            "B2_PROFILE_SCHEMA_TYPE",
        ),
        (
            "notes wrong type",
            lambda data: data.update({"notes": "not-a-list"}),
            "B2_PROFILE_SCHEMA_TYPE",
        ),
    ],
)
def test_launcher_rejects_malformed_profile_schema(
    tmp_path: Path,
    case: str,
    transform: Any,
    requirement_code: str,
) -> None:
    del case
    profile = _write_profile(tmp_path, transform)
    proc = _run_launcher(
        sys.executable,
        "-c",
        "raise SystemExit(0)",
        profile=profile,
        expected_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
    )
    _assert_cli_failure(proc, requirement_code)


@pytest.mark.parametrize(
    ("section", "setting", "value"),
    [
        ("torch", "seed", 112),
        ("torch", "use_deterministic_algorithms", False),
        ("torch", "enable_flash_sdp", True),
        ("torch", "enable_mem_efficient_sdp", True),
        ("torch", "enable_math_sdp", False),
        ("torch", "cuda.matmul.allow_tf32", True),
        ("torch", "cudnn.allow_tf32", True),
        ("torch", "cudnn.benchmark", True),
        ("torch", "cudnn.deterministic", False),
        ("torch", "float32_matmul_precision", "high"),
        ("torch", "mha_fastpath_enabled", True),
        ("model", "dtype", "float16"),
        ("model", "amp", True),
        ("model", "eval_mode", False),
        ("model", "no_grad", False),
    ],
)
def test_launcher_rejects_drift_from_each_frozen_backend_setting(
    tmp_path: Path, section: str, setting: str, value: Any
) -> None:
    profile = _write_profile(
        tmp_path,
        lambda data: data[section].update({setting: value}),
    )
    proc = _run_launcher(
        sys.executable,
        "-c",
        "raise SystemExit(0)",
        profile=profile,
        expected_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
    )
    _assert_cli_failure(proc, "B2_PROFILE_VALUE_UNSUPPORTED")


def test_launcher_rejects_invalid_required_cublas_value_even_with_matching_hash(
    tmp_path: Path,
) -> None:
    profile = _write_profile(
        tmp_path,
        lambda data: data["required_environment"].update(
            {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}
        ),
    )
    proc = _run_launcher(
        sys.executable,
        "-c",
        "raise SystemExit(0)",
        profile=profile,
        expected_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
    )
    _assert_cli_failure(proc, "B2_PROFILE_VALUE_UNSUPPORTED")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        (BOOTSTRAP_MARKER, "1"),
        (PROFILE_PATH_ENV, "/tmp/uncontrolled-profile.json"),
        (PROFILE_SHA_ENV, "0" * 64),
    ],
)
def test_launcher_rejects_inherited_contradictory_bootstrap_state(
    key: str, value: str
) -> None:
    env = _clean_env()
    env[key] = value
    proc = _run_launcher(
        sys.executable,
        "-c",
        "raise SystemExit(0)",
        env=env,
    )
    _assert_cli_failure(proc, "B2_BOOTSTRAP_STATE_CONFLICT")


def test_launcher_rejects_empty_child_command() -> None:
    proc = _run_launcher()
    _assert_cli_failure(proc, "B2_COMMAND_MISSING")


def test_launcher_does_not_run_child_after_validation_failure(tmp_path: Path) -> None:
    marker = tmp_path / "child-ran"
    child = f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')"
    proc = _run_launcher(
        sys.executable,
        "-c",
        child,
        expected_sha256="f" * 64,
    )
    _assert_cli_failure(proc, "B2_PROFILE_HASH_MISMATCH")
    assert not marker.exists()

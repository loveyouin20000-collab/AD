"""B2-00 RED: immutable runtime execution-profile attestation contract."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = REPO_ROOT / "configs" / "execution" / "frozen_deterministic_math.json"
LAUNCHER_PATH = REPO_ROOT / "tools" / "run_with_execution_profile.py"
EXPECTED_PROFILE_SHA256 = (
    "7af8dba39633743da0380fef9710940cded655f68c9efa8f84f5a52aeddb3c8d"
)
BOOTSTRAP_MARKER = "RAD_EXECUTION_PROFILE_BOOTSTRAPPED"
PROFILE_PATH_ENV = "RAD_EXECUTION_PROFILE_PATH"
PROFILE_SHA_ENV = "RAD_EXECUTION_PROFILE_SHA256"
SUBPROCESS_TIMEOUT_SECONDS = 30
ERROR_RECORD = re.compile(r"^B2_PROFILE_ERROR\[([A-Z][A-Z0-9_]*)\]: .+$")


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    for key in (BOOTSTRAP_MARKER, PROFILE_PATH_ENV, PROFILE_SHA_ENV):
        env.pop(key, None)
    return env


def _run_python(source: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", source],
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


def _assert_requirement_failure(
    proc: subprocess.CompletedProcess[str],
    expected_codes: str | set[str],
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


def _valid_attestation_process() -> subprocess.CompletedProcess[str]:
    child = r"""
import inspect
import json
from collections.abc import Mapping

from rad.runtime.execution_profile import apply_execution_profile

attestation = apply_execution_profile()
canonical = attestation.canonical_attestation()
provenance = attestation.artifact_provenance()

def json_plain_copy(value):
    if isinstance(value, Mapping):
        return {key: json_plain_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_plain_copy(item) for item in value]
    if isinstance(value, list):
        return [json_plain_copy(item) for item in value]
    return value

def mutation_is_blocked(target):
    try:
        target["b2_mutation_probe"] = True
    except (AttributeError, TypeError):
        return True
    return False

def tree_is_deeply_immutable(target):
    if hasattr(target, "keys"):
        keys = list(target.keys())
        if keys:
            key = keys[0]
            try:
                target[key] = target[key]
            except (AttributeError, TypeError):
                pass
            else:
                return False
        return all(tree_is_deeply_immutable(target[key]) for key in keys)
    if isinstance(target, (list, tuple)):
        try:
            target.append("mutation")
        except (AttributeError, TypeError):
            pass
        else:
            return False
        return all(tree_is_deeply_immutable(item) for item in target)
    return True

try:
    attestation.profile_sha256 = "changed"
except (AttributeError, TypeError):
    object_mutation_blocked = True
else:
    object_mutation_blocked = False

attestation_type = type(attestation)
signature = inspect.signature(attestation_type)
domain_values = {
    "schema_version": canonical["schema_version"],
    "profile": canonical["profile"],
    "profile_id": canonical["profile"]["profile_id"],
    "path": canonical["profile"]["path"],
    "profile_path": canonical["profile"]["path"],
    "expected_sha256": canonical["profile"]["expected_sha256"],
    "launcher_sha256": canonical["profile"]["launcher_sha256"],
    "runtime_sha256": canonical["profile"]["runtime_sha256"],
    "profile_sha256": canonical["profile"]["runtime_sha256"],
    "hashes_match": canonical["profile"]["hashes_match"],
    "requested_settings": canonical["requested_settings"],
    "effective_settings": canonical["effective_settings"],
    "environment": canonical["environment"],
    "canary": canonical["canary"],
    "attestation_sha256": attestation.attestation_sha256,
}
parameters = [
    parameter
    for name, parameter in signature.parameters.items()
    if name != "self"
    and parameter.kind
    in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY)
]
legal_values_by_parameter = [
    domain_values.get(parameter.name, canonical)
    for parameter in parameters
]
legal_kwargs = {
    parameter.name: value
    for parameter, value in zip(parameters, legal_values_by_parameter)
    if parameter.kind is not inspect.Parameter.POSITIONAL_ONLY
}
legal_positional = [
    value
    for parameter, value in zip(parameters, legal_values_by_parameter)
    if parameter.kind
    in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
]
legal_keyword_only = {
    parameter.name: value
    for parameter, value in zip(parameters, legal_values_by_parameter)
    if parameter.kind is inspect.Parameter.KEYWORD_ONLY
}

def bound_construction_is_rejected(bound_call, constructor_call):
    try:
        bound_call()
    except TypeError:
        return False
    try:
        constructor_call()
    except Exception:
        return True
    return False

kwargs_constructor_blocked = bound_construction_is_rejected(
    lambda: signature.bind(**legal_kwargs),
    lambda: attestation_type(**legal_kwargs),
)
positional_constructor_blocked = bound_construction_is_rejected(
    lambda: signature.bind(*legal_positional, **legal_keyword_only),
    lambda: attestation_type(*legal_positional, **legal_keyword_only),
)

import torch
direct_torch_observation = {
    "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    "cuda.matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
    "cudnn.benchmark": torch.backends.cudnn.benchmark,
    "cudnn.deterministic": torch.backends.cudnn.deterministic,
    "float32_matmul_precision": torch.get_float32_matmul_precision(),
    "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
    "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
    "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
}
if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "get_fastpath_enabled"):
    direct_torch_observation["mha_fastpath_enabled"] = (
        torch.backends.mha.get_fastpath_enabled()
    )

print(json.dumps({
    "canonical": json_plain_copy(canonical),
    "attestation_sha256": attestation.attestation_sha256,
    "class_name": type(attestation).__name__,
    "object_mutation_blocked": object_mutation_blocked,
    "kwargs_constructor_blocked": kwargs_constructor_blocked,
    "positional_constructor_blocked": positional_constructor_blocked,
    "requested_mutation_blocked": mutation_is_blocked(attestation.requested_settings),
    "effective_mutation_blocked": mutation_is_blocked(attestation.effective_settings),
    "provenance_mutation_blocked": mutation_is_blocked(provenance),
    "requested_environment_deep_immutable": tree_is_deeply_immutable(
        attestation.requested_settings["required_environment"]
    ),
    "requested_torch_deep_immutable": tree_is_deeply_immutable(
        attestation.requested_settings["torch"]
    ),
    "requested_model_deep_immutable": tree_is_deeply_immutable(
        attestation.requested_settings["model"]
    ),
    "effective_deep_immutable": tree_is_deeply_immutable(
        attestation.effective_settings
    ),
    "provenance_deep_immutable": tree_is_deeply_immutable(provenance),
    "direct_torch_observation": direct_torch_observation,
    "provenance": json_plain_copy(provenance),
}, sort_keys=True))
"""
    return subprocess.run(
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
            child,
        ],
        cwd=REPO_ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )


def _valid_attestation_payload() -> dict[str, Any]:
    proc = _valid_attestation_process()
    assert proc.returncode == 0, proc.stderr
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert lines, proc.stdout
    return json.loads(lines[-1])


def _bootstrapped_env(**overrides: str | None) -> dict[str, str]:
    env = _clean_env()
    env.update(
        {
            BOOTSTRAP_MARKER: "1",
            PROFILE_PATH_ENV: str(PROFILE_PATH.resolve()),
            PROFILE_SHA_ENV: EXPECTED_PROFILE_SHA256,
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


def test_frozen_profile_exact_bytes_remain_pinned() -> None:
    assert PROFILE_PATH.is_file()
    assert hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest() == EXPECTED_PROFILE_SHA256


def test_direct_runtime_application_rejects_absent_launcher_marker() -> None:
    proc = _run_python(
        "from rad.runtime.execution_profile import apply_execution_profile; "
        "apply_execution_profile()"
    )
    _assert_requirement_failure(proc, "B2_BOOTSTRAP_MARKER_MISSING")


@pytest.mark.parametrize("value", [None, "", ":16:8", ":4096:2"])
def test_runtime_rejects_absent_or_invalid_cublas_workspace_config(
    value: str | None,
) -> None:
    proc = _run_python(
        "from rad.runtime.execution_profile import apply_execution_profile; "
        "apply_execution_profile()",
        env=_bootstrapped_env(CUBLAS_WORKSPACE_CONFIG=value),
    )
    _assert_requirement_failure(proc, "B2_CUBLAS_WORKSPACE_CONFIG_INVALID")


def test_runtime_rejects_launcher_runtime_profile_hash_mismatch() -> None:
    proc = _run_python(
        "from rad.runtime.execution_profile import apply_execution_profile; "
        "apply_execution_profile()",
        env=_bootstrapped_env(
            RAD_EXECUTION_PROFILE_SHA256="0" * 64,
        ),
    )
    _assert_requirement_failure(proc, "B2_PROFILE_HASH_MISMATCH")


def test_runtime_rejects_torch_imported_before_profile_application() -> None:
    proc = _run_python(
        "import torch; "
        "from rad.runtime.execution_profile import apply_execution_profile; "
        "apply_execution_profile()",
        env=_bootstrapped_env(),
    )
    _assert_requirement_failure(proc, "B2_TORCH_PREIMPORT")


def test_runtime_rejects_prior_cuda_initialization() -> None:
    proc = _run_python(
        "import torch; "
        "torch.cuda.is_initialized = lambda: True; "
        "from rad.runtime.execution_profile import apply_execution_profile; "
        "apply_execution_profile()",
        env=_bootstrapped_env(),
    )
    _assert_requirement_failure(
        proc,
        {"B2_CUDA_ALREADY_INITIALIZED"},
    )


def test_runtime_rejects_profile_deleted_after_launcher_validation(tmp_path: Path) -> None:
    staged_profile = tmp_path / "validated-then-deleted.json"
    staged_profile.write_bytes(PROFILE_PATH.read_bytes())
    child = (
        "import os; from pathlib import Path; "
        "Path(os.environ['RAD_EXECUTION_PROFILE_PATH']).unlink(); "
        "from rad.runtime.execution_profile import apply_execution_profile; "
        "apply_execution_profile()"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER_PATH),
            "--profile",
            str(staged_profile),
            "--expected-sha256",
            EXPECTED_PROFILE_SHA256,
            "--",
            sys.executable,
            "-c",
            child,
        ],
        cwd=REPO_ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    _assert_requirement_failure(proc, "B2_PROFILE_FILE_MISSING")


def test_runtime_rereads_profile_and_rejects_toctou_content_change(
    tmp_path: Path,
) -> None:
    staged_profile = tmp_path / "validated-then-changed.json"
    staged_profile.write_bytes(PROFILE_PATH.read_bytes())
    child = (
        "import os; from pathlib import Path; "
        "p=Path(os.environ['RAD_EXECUTION_PROFILE_PATH']); "
        "p.write_bytes(p.read_bytes()+b'\\n'); "
        "assert os.environ['RAD_EXECUTION_PROFILE_SHA256'] == "
        f"{EXPECTED_PROFILE_SHA256!r}; "
        "from rad.runtime.execution_profile import apply_execution_profile; "
        "apply_execution_profile()"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER_PATH),
            "--profile",
            str(staged_profile),
            "--expected-sha256",
            EXPECTED_PROFILE_SHA256,
            "--",
            sys.executable,
            "-c",
            child,
        ],
        cwd=REPO_ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    _assert_requirement_failure(proc, "B2_PROFILE_HASH_MISMATCH")


def test_runtime_returns_only_the_controlled_attestation_type() -> None:
    payload = _valid_attestation_payload()
    assert payload["class_name"] == "ExecutionProfileAttestation"


def test_missing_attestation_cannot_be_replaced_by_public_construction() -> None:
    payload = _valid_attestation_payload()
    assert payload["kwargs_constructor_blocked"] is True
    assert payload["positional_constructor_blocked"] is True


def test_attestation_and_nested_controlled_provenance_are_immutable() -> None:
    payload = _valid_attestation_payload()
    assert payload["object_mutation_blocked"] is True
    assert payload["requested_mutation_blocked"] is True
    assert payload["effective_mutation_blocked"] is True
    assert payload["provenance_mutation_blocked"] is True
    assert payload["requested_environment_deep_immutable"] is True
    assert payload["requested_torch_deep_immutable"] is True
    assert payload["requested_model_deep_immutable"] is True
    assert payload["effective_deep_immutable"] is True
    assert payload["provenance_deep_immutable"] is True


def test_attestation_mappings_resist_dict_base_class_mutation_bypass() -> None:
    child = r"""
import hashlib
import json
from collections.abc import Mapping

from rad.runtime.execution_profile import apply_execution_profile

attestation = apply_execution_profile()

def plain(value):
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value

canonical = attestation.canonical_attestation()
canonical_before = plain(canonical)
provenance_before = plain(attestation.artifact_provenance())
attestation_hash_before = attestation.attestation_sha256
targets = {
    "canonical": canonical,
    "canonical.profile": canonical["profile"],
    "canonical.requested": canonical["requested_settings"],
    "canonical.effective": canonical["effective_settings"],
    "requested": attestation.requested_settings,
    "requested.required_environment": (
        attestation.requested_settings["required_environment"]
    ),
    "requested.torch": attestation.requested_settings["torch"],
    "requested.model": attestation.requested_settings["model"],
    "effective": attestation.effective_settings,
}
blocked = {}
for label, target in targets.items():
    try:
        dict.__setitem__(target, "__dict_base_bypass_setitem__", label)
    except TypeError:
        blocked[label + ".dict.__setitem__"] = True
    else:
        blocked[label + ".dict.__setitem__"] = False
    try:
        dict.update(target, {"__dict_base_bypass_update__": label})
    except TypeError:
        blocked[label + ".dict.update"] = True
    else:
        blocked[label + ".dict.update"] = False

canonical_after = plain(attestation.canonical_attestation())
provenance_after = plain(attestation.artifact_provenance())
attestation_hash_after = attestation.attestation_sha256
recalculated_hash = hashlib.sha256(
    json.dumps(
        canonical_after,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
).hexdigest()
print(json.dumps({
    "blocked": blocked,
    "canonical_unchanged": canonical_after == canonical_before,
    "provenance_unchanged": provenance_after == provenance_before,
    "attestation_hash_unchanged": attestation_hash_after == attestation_hash_before,
    "attestation_hash_consistent": recalculated_hash == attestation_hash_before,
}, sort_keys=True))
"""
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
            child,
        ],
        cwd=REPO_ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    payload = json.loads(lines[-1])
    assert payload["blocked"]
    assert (
        all(payload["blocked"].values())
        and payload["canonical_unchanged"] is True
        and payload["provenance_unchanged"] is True
        and payload["attestation_hash_unchanged"] is True
        and payload["attestation_hash_consistent"] is True
    ), payload


def test_canonical_attestation_has_stable_schema() -> None:
    canonical = _valid_attestation_payload()["canonical"]
    assert set(canonical) == {
        "schema_version",
        "profile",
        "requested_settings",
        "effective_settings",
        "environment",
        "canary",
    }
    assert canonical["schema_version"] == 1
    assert set(canonical["profile"]) == {
        "profile_id",
        "path",
        "expected_sha256",
        "launcher_sha256",
        "runtime_sha256",
        "hashes_match",
    }
    assert set(canonical["requested_settings"]) == {
        "required_environment",
        "torch",
        "model",
    }
    assert set(canonical["effective_settings"]) == {
        "use_deterministic_algorithms",
        "cuda.matmul.allow_tf32",
        "cudnn.allow_tf32",
        "cudnn.benchmark",
        "cudnn.deterministic",
        "float32_matmul_precision",
        "flash_sdp_enabled",
        "mem_efficient_sdp_enabled",
        "math_sdp_enabled",
        "mha_fastpath_supported",
        "mha_fastpath_enabled",
    }
    assert set(canonical["environment"]) == {
        "python_version",
        "torch_version",
        "cuda_version",
        "cudnn_version",
        "driver_version",
        "gpu_identity",
        "cuda_available",
        "cuda_initialized_before_apply",
        "cublas_workspace_config",
    }
    assert set(canonical["canary"]) == {
        "self_repeatability",
        "independent_reconstruction",
        "execution_count",
        "reconstruction_count",
        "first_output_sha256",
        "repeat_output_sha256",
        "reconstructed_output_sha256",
        "first_input_sha256",
        "reconstructed_input_sha256",
        "first_module_sha256",
        "reconstructed_module_sha256",
        "first_input_identity",
        "reconstructed_input_identity",
        "first_module_identity",
        "reconstructed_module_identity",
        "input_rebuilt",
        "module_rebuilt",
    }


def test_attestation_profile_identity_and_hash_agreement_are_observed() -> None:
    profile = _valid_attestation_payload()["canonical"]["profile"]
    assert profile == {
        "profile_id": "frozen_deterministic_math",
        "path": str(PROFILE_PATH.resolve()),
        "expected_sha256": EXPECTED_PROFILE_SHA256,
        "launcher_sha256": EXPECTED_PROFILE_SHA256,
        "runtime_sha256": EXPECTED_PROFILE_SHA256,
        "hashes_match": True,
    }


def test_requested_settings_cover_the_complete_frozen_profile() -> None:
    requested = _valid_attestation_payload()["canonical"]["requested_settings"]
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    assert requested == {
        "required_environment": profile["required_environment"],
        "torch": profile["torch"],
        "model": profile["model"],
    }
    assert set(requested["required_environment"]) == set(profile["required_environment"])
    assert set(requested["torch"]) == set(profile["torch"])
    assert set(requested["model"]) == set(profile["model"])


def test_effective_settings_attest_every_frozen_backend_control() -> None:
    effective = _valid_attestation_payload()["canonical"]["effective_settings"]
    assert effective["use_deterministic_algorithms"] is True
    assert effective["cuda.matmul.allow_tf32"] is False
    assert effective["cudnn.allow_tf32"] is False
    assert effective["cudnn.benchmark"] is False
    assert effective["cudnn.deterministic"] is True
    assert effective["float32_matmul_precision"] == "highest"
    assert effective["flash_sdp_enabled"] is False
    assert effective["mem_efficient_sdp_enabled"] is False
    assert effective["math_sdp_enabled"] is True
    if effective["mha_fastpath_supported"]:
        assert effective["mha_fastpath_enabled"] is False


def test_torch_getters_confirm_effective_backend_instead_of_trusting_attestation() -> None:
    observed = _valid_attestation_payload()["direct_torch_observation"]
    assert observed["use_deterministic_algorithms"] is True
    assert observed["cuda.matmul.allow_tf32"] is False
    assert observed["cudnn.allow_tf32"] is False
    assert observed["cudnn.benchmark"] is False
    assert observed["cudnn.deterministic"] is True
    assert observed["float32_matmul_precision"] == "highest"
    assert observed["flash_sdp_enabled"] is False
    assert observed["mem_efficient_sdp_enabled"] is False
    assert observed["math_sdp_enabled"] is True
    if "mha_fastpath_enabled" in observed:
        assert observed["mha_fastpath_enabled"] is False


def test_environment_attestation_has_stable_runtime_identity_fields() -> None:
    environment = _valid_attestation_payload()["canonical"]["environment"]
    assert set(environment) == {
        "python_version",
        "torch_version",
        "cuda_version",
        "cudnn_version",
        "driver_version",
        "gpu_identity",
        "cuda_available",
        "cuda_initialized_before_apply",
        "cublas_workspace_config",
    }
    assert environment["cuda_initialized_before_apply"] is False
    assert environment["cublas_workspace_config"] == ":4096:8"


def test_canary_proves_same_runtime_self_repeatability() -> None:
    canary = _valid_attestation_payload()["canonical"]["canary"]
    assert canary["self_repeatability"] is True
    assert canary["execution_count"] == 3
    assert canary["first_output_sha256"] == canary["repeat_output_sha256"]


def test_canary_proves_independently_reconstructed_execution() -> None:
    canary = _valid_attestation_payload()["canonical"]["canary"]
    assert canary["independent_reconstruction"] is True
    assert canary["reconstruction_count"] == 1
    assert canary["input_rebuilt"] is True
    assert canary["module_rebuilt"] is True
    assert canary["first_input_sha256"] == canary["reconstructed_input_sha256"]
    assert canary["first_module_sha256"] == canary["reconstructed_module_sha256"]
    assert canary["first_input_identity"] != canary["reconstructed_input_identity"]
    assert canary["first_module_identity"] != canary["reconstructed_module_identity"]
    assert canary["first_output_sha256"] == canary["reconstructed_output_sha256"]


def test_attention_canary_reconstructs_module_and_inputs_in_real_execution() -> None:
    child = r"""
import builtins
import json
import sys

real_import = builtins.__import__
records = {"constructed_modules": [], "forwards": []}
retained_objects = []
state = {"loading_torch": False, "patched": False}

def import_with_mha_observer(name, globals=None, locals=None, fromlist=(), level=0):
    outer_torch_import = name == "torch" and not state["loading_torch"]
    if outer_torch_import:
        state["loading_torch"] = True
    try:
        module = real_import(name, globals, locals, fromlist, level)
    finally:
        if outer_torch_import:
            state["loading_torch"] = False
    if outer_torch_import and not state["patched"]:
        torch = sys.modules["torch"]
        original_mha = torch.nn.MultiheadAttention

        class ObservedMultiheadAttention(original_mha):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                records["constructed_modules"].append(id(self))
                retained_objects.append(self)

            def forward(self, query, key, value, *args, **kwargs):
                records["forwards"].append({
                    "module": id(self),
                    "query": id(query),
                    "key": id(key),
                    "value": id(value),
                })
                retained_objects.extend((query, key, value))
                return super().forward(query, key, value, *args, **kwargs)

        torch.nn.MultiheadAttention = ObservedMultiheadAttention
        state["patched"] = True
    return module

builtins.__import__ = import_with_mha_observer
import rad.runtime.execution_profile as execution_profile
if "torch" in sys.modules:
    raise RuntimeError("runtime imported torch before apply_execution_profile entry")
attestation = execution_profile.apply_execution_profile()
records["attestation_sha256"] = attestation.attestation_sha256
print(json.dumps(records, sort_keys=True))
"""
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
            child,
        ],
        cwd=REPO_ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    records = json.loads(lines[-1])
    modules = records["constructed_modules"]
    forwards = records["forwards"]
    assert len(modules) == 2
    assert len(forwards) == 3
    assert [item["module"] for item in forwards] == [modules[0], modules[0], modules[1]]
    for tensor_name in ("query", "key", "value"):
        assert forwards[0][tensor_name] == forwards[1][tensor_name]
        assert forwards[2][tensor_name] != forwards[0][tensor_name]


@pytest.mark.parametrize(
    ("perturbed_call", "requirement_code"),
    [
        (2, "B2_CANARY_SELF_REPEATABILITY_FAILED"),
        (3, "B2_CANARY_INDEPENDENT_RECONSTRUCTION_FAILED"),
    ],
)
def test_attention_canary_mismatch_fails_closed_without_preimporting_torch(
    perturbed_call: int,
    requirement_code: str,
) -> None:
    child = f"""
import builtins
import sys

real_import = builtins.__import__
state = {{"loading_torch": False, "patched": False, "forward_calls": 0}}

def import_with_mha_fault(name, globals=None, locals=None, fromlist=(), level=0):
    outer_torch_import = name == "torch" and not state["loading_torch"]
    if outer_torch_import:
        state["loading_torch"] = True
    try:
        module = real_import(name, globals, locals, fromlist, level)
    finally:
        if outer_torch_import:
            state["loading_torch"] = False
    if outer_torch_import and not state["patched"]:
        torch = sys.modules["torch"]
        original_forward = torch.nn.MultiheadAttention.forward

        def perturbed_forward(self, *args, **kwargs):
            result = original_forward(self, *args, **kwargs)
            state["forward_calls"] += 1
            if state["forward_calls"] != {perturbed_call}:
                return result
            if isinstance(result, tuple):
                changed = result[0] + torch.ones_like(result[0])
                return (changed, *result[1:])
            return result + torch.ones_like(result)

        torch.nn.MultiheadAttention.forward = perturbed_forward
        state["patched"] = True
    return module

builtins.__import__ = import_with_mha_fault
import rad.runtime.execution_profile as execution_profile
if "torch" in sys.modules:
    raise RuntimeError("runtime imported torch before apply_execution_profile entry")
attestation = execution_profile.apply_execution_profile()
print("ATTESTATION_CREATED", attestation.attestation_sha256)
"""
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
            child,
        ],
        cwd=REPO_ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    _assert_requirement_failure(proc, requirement_code)
    assert "ATTESTATION_CREATED" not in proc.stdout


def test_attestation_sha256_is_canonical_and_stable_across_fresh_processes() -> None:
    first = _valid_attestation_payload()
    canonical_bytes = json.dumps(
        first["canonical"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    assert first["attestation_sha256"] == hashlib.sha256(canonical_bytes).hexdigest()

    second = _valid_attestation_payload()
    assert second["canonical"] == first["canonical"]
    assert second["attestation_sha256"] == first["attestation_sha256"]


def test_artifact_provenance_is_derived_from_attestation_and_not_user_supplied() -> None:
    payload = _valid_attestation_payload()
    provenance = payload["provenance"]
    assert set(provenance) == {
        "execution_profile_name",
        "execution_profile_sha256",
        "runtime_attestation_sha256",
    }
    assert provenance == {
        "execution_profile_name": "frozen_deterministic_math",
        "execution_profile_sha256": EXPECTED_PROFILE_SHA256,
        "runtime_attestation_sha256": payload["attestation_sha256"],
    }


def test_attestation_issuer_is_not_exposed_and_forged_instance_is_rejected() -> None:
    module = importlib.import_module("rad.runtime.execution_profile")
    assert not hasattr(module, "_issue_attestation")
    forged = object.__new__(module.ExecutionProfileAttestation)
    assert module.is_controlled_execution_profile_attestation(forged) is False

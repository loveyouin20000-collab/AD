"""Apply and attest the frozen deterministic execution profile.

This module deliberately has a standard-library-only import surface.  Torch is
first imported inside :func:`apply_execution_profile`, after launcher state and
the process pre-import boundary have been checked.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, NoReturn

BOOTSTRAP_MARKER = "RAD_EXECUTION_PROFILE_BOOTSTRAPPED"
PROFILE_PATH_ENV = "RAD_EXECUTION_PROFILE_PATH"
PROFILE_SHA_ENV = "RAD_EXECUTION_PROFILE_SHA256"
APPROVED_PROFILE_SHA256 = (
    "7af8dba39633743da0380fef9710940cded655f68c9efa8f84f5a52aeddb3c8d"
)
REQUIRED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


class _ProfileError(RuntimeError):
    """A structured, fail-closed runtime profile error."""


def _fail(code: str, detail: str) -> NoReturn:
    message = f"B2_PROFILE_ERROR[{code}]: {detail}"
    print(message, file=sys.stderr)
    raise _ProfileError(message)


class _ImmutableMapping(Mapping[str, Any]):
    """A mapping backed only by recursively immutable tuples."""

    __slots__ = ("_items",)

    _items: tuple[tuple[str, Any], ...]

    def __init__(self, items: tuple[tuple[str, Any], ...]) -> None:
        object.__setattr__(self, "_items", items)

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        del name, value
        raise TypeError("controlled attestation mappings are immutable")

    def __getitem__(self, key: str) -> Any:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _ImmutableMapping(
            tuple((key, _deep_freeze(item)) for key, item in value.items())
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _thaw_for_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_for_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_for_json(item) for item in value]
    return value


class ExecutionProfileAttestation:
    """Immutable evidence issued only by successful runtime profile application."""

    _canonical: Mapping[str, Any]
    _control_seal: object
    attestation_sha256: str
    effective_settings: Mapping[str, Any]
    requested_settings: Mapping[str, Any]

    __slots__ = (
        "_canonical",
        "_control_seal",
        "attestation_sha256",
        "effective_settings",
        "requested_settings",
    )

    def __new__(cls) -> ExecutionProfileAttestation:
        del cls
        raise TypeError("ExecutionProfileAttestation is issued by apply_execution_profile()")

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        del name, value
        raise TypeError("ExecutionProfileAttestation is immutable")

    def canonical_attestation(self) -> Mapping[str, Any]:
        """Return the deeply immutable canonical attestation tree."""
        return self._canonical

    def artifact_provenance(self) -> Mapping[str, str]:
        """Derive the only artifact provenance fields exposed to callers."""
        profile = self._canonical["profile"]
        return _deep_freeze(
            {
                "execution_profile_name": profile["profile_id"],
                "execution_profile_sha256": profile["runtime_sha256"],
                "runtime_attestation_sha256": self.attestation_sha256,
            }
        )

def _read_profile(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        _fail("B2_PROFILE_FILE_MISSING", f"profile file does not exist: {path}")
    except OSError as exc:
        _fail("B2_PROFILE_FILE_UNREADABLE", f"cannot read profile file: {exc}")

    runtime_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        profile = json.loads(raw)
    except (json.JSONDecodeError, UnicodeError) as exc:
        _fail("B2_PROFILE_JSON_INVALID", f"profile is not valid JSON: {exc}")
    if not isinstance(profile, dict):
        _fail("B2_PROFILE_SCHEMA_TYPE", "profile must be a JSON object")
    return profile, runtime_sha256


def _validate_bootstrap() -> tuple[Path, str, dict[str, Any], str]:
    if os.environ.get(BOOTSTRAP_MARKER) != "1":
        _fail("B2_BOOTSTRAP_MARKER_MISSING", "validated launcher marker is absent")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != REQUIRED_CUBLAS_WORKSPACE_CONFIG:
        _fail(
            "B2_CUBLAS_WORKSPACE_CONFIG_INVALID",
            f"CUBLAS_WORKSPACE_CONFIG must be {REQUIRED_CUBLAS_WORKSPACE_CONFIG}",
        )

    profile_path_text = os.environ.get(PROFILE_PATH_ENV)
    launcher_sha256 = os.environ.get(PROFILE_SHA_ENV)
    if not profile_path_text:
        _fail("B2_PROFILE_PATH_MISSING", "validated launcher profile path is absent")
    if not launcher_sha256:
        _fail("B2_PROFILE_HASH_MISSING", "validated launcher profile hash is absent")

    profile_path = Path(profile_path_text).resolve()
    profile, runtime_sha256 = _read_profile(profile_path)
    if (
        launcher_sha256 != runtime_sha256
        or runtime_sha256 != APPROVED_PROFILE_SHA256
    ):
        _fail(
            "B2_PROFILE_HASH_MISMATCH",
            f"launcher={launcher_sha256}, runtime={runtime_sha256}",
        )
    return profile_path, launcher_sha256, profile, runtime_sha256


def _apply_backend_settings(torch: Any, settings: Mapping[str, Any]) -> dict[str, Any]:
    torch.use_deterministic_algorithms(settings["use_deterministic_algorithms"])
    torch.backends.cuda.matmul.allow_tf32 = settings["cuda.matmul.allow_tf32"]
    torch.backends.cudnn.allow_tf32 = settings["cudnn.allow_tf32"]
    torch.backends.cudnn.benchmark = settings["cudnn.benchmark"]
    torch.backends.cudnn.deterministic = settings["cudnn.deterministic"]
    torch.set_float32_matmul_precision(settings["float32_matmul_precision"])
    torch.backends.cuda.enable_flash_sdp(settings["enable_flash_sdp"])
    torch.backends.cuda.enable_mem_efficient_sdp(
        settings["enable_mem_efficient_sdp"]
    )
    torch.backends.cuda.enable_math_sdp(settings["enable_math_sdp"])

    mha_backend = getattr(torch.backends, "mha", None)
    mha_supported = bool(
        mha_backend is not None
        and hasattr(mha_backend, "set_fastpath_enabled")
        and hasattr(mha_backend, "get_fastpath_enabled")
    )
    if mha_supported:
        assert mha_backend is not None
        mha_backend.set_fastpath_enabled(settings["mha_fastpath_enabled"])

    mha_fastpath_enabled = None
    if mha_supported:
        assert mha_backend is not None
        mha_fastpath_enabled = mha_backend.get_fastpath_enabled()
    effective = {
        "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda.matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn.benchmark": torch.backends.cudnn.benchmark,
        "cudnn.deterministic": torch.backends.cudnn.deterministic,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
        "mha_fastpath_supported": mha_supported,
        "mha_fastpath_enabled": mha_fastpath_enabled,
    }
    expected = {
        "use_deterministic_algorithms": settings["use_deterministic_algorithms"],
        "cuda.matmul.allow_tf32": settings["cuda.matmul.allow_tf32"],
        "cudnn.allow_tf32": settings["cudnn.allow_tf32"],
        "cudnn.benchmark": settings["cudnn.benchmark"],
        "cudnn.deterministic": settings["cudnn.deterministic"],
        "float32_matmul_precision": settings["float32_matmul_precision"],
        "flash_sdp_enabled": settings["enable_flash_sdp"],
        "mem_efficient_sdp_enabled": settings["enable_mem_efficient_sdp"],
        "math_sdp_enabled": settings["enable_math_sdp"],
    }
    if mha_supported:
        expected["mha_fastpath_enabled"] = settings["mha_fastpath_enabled"]
    mismatches = [
        key for key, expected_value in expected.items()
        if effective[key] != expected_value
    ]
    if mismatches:
        _fail(
            "B2_BACKEND_SETTING_INEFFECTIVE",
            f"torch getter disagrees for {mismatches[0]}",
        )
    return effective


def _tensor_sha256(tensor: Any) -> str:
    data = tensor.detach().cpu().contiguous().view(-1).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _module_sha256(module: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.detach().cpu().contiguous().view(-1).numpy().tobytes())
    return digest.hexdigest()


def _build_canary_case(torch: Any, seed: int) -> tuple[Any, Any]:
    torch.manual_seed(seed)
    module = torch.nn.MultiheadAttention(
        embed_dim=8,
        num_heads=2,
        dropout=0.0,
        batch_first=True,
        device="cpu",
        dtype=torch.float32,
    )
    module.eval()
    query = torch.randn(2, 4, 8, device="cpu", dtype=torch.float32)
    return module, query


def _run_canary(torch: Any, seed: int, configured_rng_state: Any) -> dict[str, Any]:
    try:
        first_module, first_input = _build_canary_case(torch, seed)
        with torch.no_grad():
            first_output = first_module(first_input, first_input, first_input)[0]
            repeat_output = first_module(first_input, first_input, first_input)[0]

        first_output_sha256 = _tensor_sha256(first_output)
        repeat_output_sha256 = _tensor_sha256(repeat_output)
        self_repeatability = bool(
            torch.equal(first_output, repeat_output)
            and first_output_sha256 == repeat_output_sha256
        )
        if not self_repeatability:
            _fail(
                "B2_CANARY_SELF_REPEATABILITY_FAILED",
                "repeated attention output was not bitwise identical",
            )

        reconstructed_module, reconstructed_input = _build_canary_case(torch, seed)
        with torch.no_grad():
            reconstructed_output = reconstructed_module(
                reconstructed_input,
                reconstructed_input,
                reconstructed_input,
            )[0]

        reconstructed_output_sha256 = _tensor_sha256(reconstructed_output)
        first_input_sha256 = _tensor_sha256(first_input)
        reconstructed_input_sha256 = _tensor_sha256(reconstructed_input)
        first_module_sha256 = _module_sha256(first_module)
        reconstructed_module_sha256 = _module_sha256(reconstructed_module)
        independent_reconstruction = bool(
            torch.equal(first_output, reconstructed_output)
            and first_output_sha256 == reconstructed_output_sha256
            and first_input_sha256 == reconstructed_input_sha256
            and first_module_sha256 == reconstructed_module_sha256
        )
        if not independent_reconstruction:
            _fail(
                "B2_CANARY_INDEPENDENT_RECONSTRUCTION_FAILED",
                "reconstructed attention execution was not bitwise identical",
            )
        return {
            "self_repeatability": self_repeatability,
            "independent_reconstruction": independent_reconstruction,
            "execution_count": 3,
            "reconstruction_count": 1,
            "first_output_sha256": first_output_sha256,
            "repeat_output_sha256": repeat_output_sha256,
            "reconstructed_output_sha256": reconstructed_output_sha256,
            "first_input_sha256": first_input_sha256,
            "reconstructed_input_sha256": reconstructed_input_sha256,
            "first_module_sha256": first_module_sha256,
            "reconstructed_module_sha256": reconstructed_module_sha256,
            "first_input_identity": "input_instance_1",
            "reconstructed_input_identity": "input_instance_2",
            "first_module_identity": "module_instance_1",
            "reconstructed_module_identity": "module_instance_2",
            "input_rebuilt": first_input is not reconstructed_input,
            "module_rebuilt": first_module is not reconstructed_module,
        }
    except _ProfileError:
        raise
    except Exception as exc:
        _fail("B2_CANARY_EXECUTION_FAILED", f"attention canary failed: {exc}")
    finally:
        torch.random.set_rng_state(configured_rng_state)


def _driver_version(torch: Any) -> str | None:
    getter = getattr(torch._C, "_cuda_getDriverVersion", None)
    if getter is None:
        return None
    try:
        value = getter()
    except Exception:
        return None
    return str(value)


def _gpu_identity(torch: Any, cuda_available: bool) -> list[dict[str, Any]]:
    if not cuda_available:
        return []
    identities = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        identities.append(
            {
                "index": index,
                "name": properties.name,
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory": properties.total_memory,
                "uuid": str(getattr(properties, "uuid", "")) or None,
            }
        )
    return identities


def _apply_execution_profile_impl(
    issue_attestation: Callable[[dict[str, Any]], ExecutionProfileAttestation],
) -> ExecutionProfileAttestation:
    torch_preimported = "torch" in sys.modules
    cuda_initialized_before_apply = False
    if torch_preimported:
        preimported_torch = sys.modules["torch"]
        try:
            cuda_initialized_before_apply = bool(
                preimported_torch.cuda.is_initialized()
            )
        except Exception:
            cuda_initialized_before_apply = False
        if cuda_initialized_before_apply:
            _fail(
                "B2_CUDA_ALREADY_INITIALIZED",
                "CUDA was initialized before execution-profile application",
            )
        _fail(
            "B2_TORCH_PREIMPORT",
            "torch was imported before execution-profile application",
        )

    profile_path, launcher_sha256, profile, runtime_sha256 = _validate_bootstrap()

    try:
        import torch
    except Exception as exc:
        _fail("B2_TORCH_IMPORT_FAILED", f"cannot import torch: {exc}")

    torch_settings = profile["torch"]
    torch.manual_seed(torch_settings["seed"])
    configured_rng_state = torch.random.get_rng_state().clone()
    effective_settings = _apply_backend_settings(torch, torch_settings)
    canary = _run_canary(torch, torch_settings["seed"], configured_rng_state)

    cuda_available = bool(torch.cuda.is_available())
    environment = {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "driver_version": _driver_version(torch),
        "gpu_identity": _gpu_identity(torch, cuda_available),
        "cuda_available": cuda_available,
        "cuda_initialized_before_apply": cuda_initialized_before_apply,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    requested_settings = {
        "required_environment": profile["required_environment"],
        "torch": torch_settings,
        "model": profile["model"],
    }
    canonical = {
        "schema_version": 1,
        "profile": {
            "profile_id": profile["profile_id"],
            "path": str(profile_path),
            "expected_sha256": APPROVED_PROFILE_SHA256,
            "launcher_sha256": launcher_sha256,
            "runtime_sha256": runtime_sha256,
            "hashes_match": launcher_sha256 == runtime_sha256,
        },
        "requested_settings": requested_settings,
        "effective_settings": effective_settings,
        "environment": environment,
        "canary": canary,
    }
    return issue_attestation(canonical)


def _build_execution_profile_api() -> tuple[
    Callable[[], ExecutionProfileAttestation],
    Callable[[object], bool],
]:
    control_seal = object()
    implementation = _apply_execution_profile_impl

    def issue_attestation(
        canonical: dict[str, Any],
    ) -> ExecutionProfileAttestation:
        frozen = _deep_freeze(canonical)
        canonical_bytes = json.dumps(
            _thaw_for_json(frozen),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        attestation = object.__new__(ExecutionProfileAttestation)
        object.__setattr__(attestation, "_canonical", frozen)
        object.__setattr__(attestation, "_control_seal", control_seal)
        object.__setattr__(
            attestation,
            "attestation_sha256",
            hashlib.sha256(canonical_bytes).hexdigest(),
        )
        object.__setattr__(
            attestation,
            "requested_settings",
            frozen["requested_settings"],
        )
        object.__setattr__(
            attestation,
            "effective_settings",
            frozen["effective_settings"],
        )
        return attestation

    def apply_execution_profile() -> ExecutionProfileAttestation:
        """Apply the launcher-validated profile and return controlled evidence."""
        return implementation(issue_attestation)

    def is_controlled_execution_profile_attestation(value: object) -> bool:
        """Return whether value carries this process's private control seal."""
        return (
            type(value) is ExecutionProfileAttestation
            and getattr(value, "_control_seal", None) is control_seal
        )

    return apply_execution_profile, is_controlled_execution_profile_attestation


(
    apply_execution_profile,
    is_controlled_execution_profile_attestation,
) = _build_execution_profile_api()
globals().pop("_apply_execution_profile_impl")
globals().pop("_build_execution_profile_api")

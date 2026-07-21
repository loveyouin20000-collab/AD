#!/usr/bin/env python3
"""Validate the frozen execution profile before launching a child process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

BOOTSTRAP_MARKER = "RAD_EXECUTION_PROFILE_BOOTSTRAPPED"
PROFILE_PATH_ENV = "RAD_EXECUTION_PROFILE_PATH"
PROFILE_SHA_ENV = "RAD_EXECUTION_PROFILE_SHA256"
BOOTSTRAP_KEYS = (BOOTSTRAP_MARKER, PROFILE_PATH_ENV, PROFILE_SHA_ENV)
APPROVED_PROFILE_SHA256 = (
    "7af8dba39633743da0380fef9710940cded655f68c9efa8f84f5a52aeddb3c8d"
)
READ_CHUNK_SIZE = 1024 * 1024

FROZEN_PROFILE: dict[str, Any] = {
    "schema_version": 1,
    "profile_id": "frozen_deterministic_math",
    "applies_to": ["B1", "B2", "paper_experiments"],
    "required_environment": {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    },
    "torch": {
        "seed": 111,
        "use_deterministic_algorithms": True,
        "cuda.matmul.allow_tf32": False,
        "cudnn.allow_tf32": False,
        "cudnn.benchmark": False,
        "cudnn.deterministic": True,
        "float32_matmul_precision": "highest",
        "enable_flash_sdp": False,
        "enable_mem_efficient_sdp": False,
        "enable_math_sdp": True,
        "mha_fastpath_enabled": False,
    },
    "model": {
        "dtype": "float32",
        "amp": False,
        "eval_mode": True,
        "no_grad": True,
    },
    "notes": [
        (
            "Selected by B1-05 backend matrix: production_default_attention is not "
            "independently deterministic."
        ),
        (
            "strict_independent_pass is valid only while this profile remains the "
            "project-wide B2/paper constraint."
        ),
    ],
}


class ProfileError(Exception):
    """A fail-closed launcher contract violation."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise ProfileError(code, detail)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a command under a validated frozen execution profile."
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    return args


def _read_and_hash(path: Path) -> tuple[bytes, str]:
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(READ_CHUNK_SIZE):
                digest.update(chunk)
                chunks.append(chunk)
    except FileNotFoundError:
        _fail("B2_PROFILE_FILE_MISSING", f"profile file does not exist: {path}")
    except OSError as exc:
        _fail("B2_PROFILE_FILE_UNREADABLE", f"cannot read profile file: {exc}")
    return b"".join(chunks), digest.hexdigest()


def _validate_frozen_node(actual: Any, expected: Any, location: str) -> None:
    if type(actual) is not type(expected):
        _fail(
            "B2_PROFILE_SCHEMA_TYPE",
            f"{location} must have type {type(expected).__name__}",
        )

    if isinstance(expected, dict):
        missing = [key for key in expected if key not in actual]
        if missing:
            _fail(
                "B2_PROFILE_SCHEMA_MISSING",
                f"{location} is missing key {missing[0]!r}",
            )
        unknown = [key for key in actual if key not in expected]
        if unknown:
            _fail(
                "B2_PROFILE_SCHEMA_UNKNOWN",
                f"{location} contains unknown key {unknown[0]!r}",
            )
        for key, expected_value in expected.items():
            _validate_frozen_node(actual[key], expected_value, f"{location}.{key}")
        return

    if isinstance(expected, list):
        if len(actual) != len(expected):
            _fail(
                "B2_PROFILE_VALUE_UNSUPPORTED",
                f"{location} must contain exactly {len(expected)} elements",
            )
        for index, expected_value in enumerate(expected):
            _validate_frozen_node(actual[index], expected_value, f"{location}[{index}]")
        return

    if actual != expected:
        _fail(
            "B2_PROFILE_VALUE_UNSUPPORTED",
            f"{location} differs from the frozen value",
        )


def _load_and_validate_profile(path: Path, expected_sha256: str) -> str:
    raw_profile, actual_sha256 = _read_and_hash(path)
    if actual_sha256 != expected_sha256:
        _fail(
            "B2_PROFILE_HASH_MISMATCH",
            f"expected SHA-256 {expected_sha256}, observed {actual_sha256}",
        )
    try:
        profile = json.loads(raw_profile)
    except (json.JSONDecodeError, UnicodeError) as exc:
        _fail("B2_PROFILE_JSON_INVALID", f"profile is not valid JSON: {exc}")
    _validate_frozen_node(profile, FROZEN_PROFILE, "profile")
    if actual_sha256 != APPROVED_PROFILE_SHA256:
        _fail(
            "B2_PROFILE_IDENTITY_NOT_APPROVED",
            f"profile SHA-256 is not approved: {actual_sha256}",
        )
    return actual_sha256


def _reject_inherited_bootstrap_state() -> None:
    inherited = [key for key in BOOTSTRAP_KEYS if key in os.environ]
    if inherited:
        _fail(
            "B2_BOOTSTRAP_STATE_CONFLICT",
            f"inherited bootstrap variable is forbidden: {inherited[0]}",
        )


def _run(argv: list[str]) -> int:
    args = _parse_args(argv)
    _reject_inherited_bootstrap_state()
    if not args.command:
        _fail("B2_COMMAND_MISSING", "a child command is required after --")

    profile_path = Path(args.profile).resolve()
    profile_sha256 = _load_and_validate_profile(
        profile_path,
        str(args.expected_sha256),
    )

    child_environment = os.environ.copy()
    child_environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    child_environment[BOOTSTRAP_MARKER] = "1"
    child_environment[PROFILE_PATH_ENV] = str(profile_path)
    child_environment[PROFILE_SHA_ENV] = profile_sha256
    completed = subprocess.run(
        list(args.command),
        env=child_environment,
        check=False,
    )
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    """Run the launcher and emit one structured record for contract failures."""
    try:
        return _run(sys.argv[1:] if argv is None else argv)
    except ProfileError as exc:
        print(f"B2_PROFILE_ERROR[{exc.code}]: {exc.detail}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

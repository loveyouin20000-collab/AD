"""Atomic artifact writing and output-directory protection."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from rad.errors import OutputProtectionError


def atomic_write_json(path: Path | str, payload: Any) -> None:
    """Write JSON via temp file → flush → fsync → os.replace."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, destination)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def refuse_existing_run(path: Path | str) -> None:
    """Refuse silent overwrite of an existing run path."""
    target = Path(path)
    if target.exists():
        raise OutputProtectionError(
            f"Refusing to overwrite existing run path: {target}"
        )

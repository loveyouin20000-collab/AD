"""Deprecated entrypoint: synthetic adaptive smoke moved to smoke_adaptive_engine.py.

For paper / dataset-backed evaluation use tools/evaluate_adaptive_dataset.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.smoke_adaptive_engine import (  # noqa: E402
    build_engine,
    load_profile,
    main,
)


if __name__ == "__main__":
    print(
        "WARNING: tools/evaluate_adaptive.py is a compatibility shim for "
        "tools/smoke_adaptive_engine.py (synthetic smoke only). "
        "Use tools/evaluate_adaptive_dataset.py for real-dataset paper evaluation.",
        file=sys.stderr,
    )
    raise SystemExit(main())

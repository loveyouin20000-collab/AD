"""B1-05: frozen execution profile hash and release-closure selection contracts."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = REPO_ROOT / "configs" / "execution" / "frozen_deterministic_math.json"
EXPECTED_PROFILE_SHA256 = (
    "7af8dba39633743da0380fef9710940cded655f68c9efa8f84f5a52aeddb3c8d"
)
RELEASE_CLOSURE = REPO_ROOT / "tools" / "b1_05_release_closure.py"


def test_frozen_deterministic_math_profile_hash_is_pinned() -> None:
    assert PROFILE_PATH.is_file()
    digest = hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest()
    assert digest == EXPECTED_PROFILE_SHA256


def test_release_closure_selects_frozen_profile_when_production_is_noisy() -> None:
    spec = importlib.util.spec_from_file_location(
        "b1_05_release_closure", RELEASE_CLOSURE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    matrix = {
        "profiles": {
            "frozen_deterministic_math": {
                "official_self_max": 0.0,
                "staged_self_max": 0.0,
                "cross_path_max": 4.77e-07,
            },
            "production_default_attention": {
                "official_self_max": 4.72e-05,
                "staged_self_max": 6.96e-05,
                "cross_path_max": 6.63e-05,
            },
        }
    }
    decision = module._decide_backend(matrix)
    assert decision["selected_b2_profile"] == "frozen_deterministic_math"
    assert decision["pass_detail_candidate"] == "strict_independent_pass"
    assert decision["requires_project_wide_freeze"] is True
    assert decision["production_envelope"]["envelope_ok"] is True


def test_release_closure_blocks_when_no_profile_is_deterministic() -> None:
    spec = importlib.util.spec_from_file_location(
        "b1_05_release_closure", RELEASE_CLOSURE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    matrix = {
        "profiles": {
            "frozen_deterministic_math": {
                "official_self_max": 1e-4,
                "staged_self_max": 1e-4,
                "cross_path_max": 2e-4,
            },
            "production_default_attention": {
                "official_self_max": 1e-4,
                "staged_self_max": 1e-4,
                "cross_path_max": 2e-4,
            },
        }
    }
    decision = module._decide_backend(matrix)
    assert decision["selected_b2_profile"] is None
    assert "blocked_reason" in decision

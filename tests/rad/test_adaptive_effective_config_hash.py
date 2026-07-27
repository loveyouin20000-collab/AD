from __future__ import annotations

import textwrap
from pathlib import Path

import yaml

from rad.evaluation.effective_config import (
    adaptive_config_identity,
    canonical_config_sha256,
    deep_merge_config,
    load_yaml_mapping,
)

REPO = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO / "configs" / "rad" / "adaptive.yaml"
OVERLAY_A = REPO / "configs" / "rad" / "matrix" / "fixed_exit_12_equal.yaml"
OVERLAY_B = REPO / "configs" / "rad" / "matrix" / "fixed_exit_18_equal.yaml"


def _write_overlay(path: Path, *, exit_depth: int) -> None:
    path.write_text(
        textwrap.dedent(
            f"""\
            method:
              name: fixed_exit
              exit_depth: {exit_depth}
              fusion: equal
            adaptive:
              fixed_exit_depth: {exit_depth}
              fusion_mode: equal
              early_depths: []
              full_depth: {exit_depth}
            """
        ),
        encoding="utf-8",
    )


def test_same_base_different_overlays_share_base_hash_only() -> None:
    base_only = adaptive_config_identity(BASE_CONFIG, overlay_path=None)
    with_a = adaptive_config_identity(BASE_CONFIG, overlay_path=OVERLAY_A)
    with_b = adaptive_config_identity(BASE_CONFIG, overlay_path=OVERLAY_B)

    assert with_a.base_config_sha256 == with_b.base_config_sha256 == base_only.base_config_sha256
    assert with_a.overlay_sha256 == adaptive_config_identity(
        BASE_CONFIG, overlay_path=OVERLAY_A
    ).overlay_sha256
    assert with_b.overlay_sha256 == adaptive_config_identity(
        BASE_CONFIG, overlay_path=OVERLAY_B
    ).overlay_sha256
    assert with_a.overlay_sha256 != with_b.overlay_sha256
    assert with_a.effective_config_sha256 != with_b.effective_config_sha256
    assert with_a.effective_config_sha256 != base_only.effective_config_sha256


def test_reapplying_same_overlay_is_idempotent_for_effective_hash(tmp_path: Path) -> None:
    overlay_copy = tmp_path / "overlay_copy.yaml"
    overlay_copy.write_bytes(OVERLAY_A.read_bytes())

    first = adaptive_config_identity(BASE_CONFIG, overlay_path=OVERLAY_A)
    second = adaptive_config_identity(BASE_CONFIG, overlay_path=overlay_copy)

    assert first.overlay_sha256 == second.overlay_sha256
    assert first.effective_config_sha256 == second.effective_config_sha256


def test_effective_hash_is_independent_of_yaml_formatting(tmp_path: Path) -> None:
    overlay_compact = tmp_path / "compact.yaml"
    overlay_spaced = tmp_path / "spaced.yaml"
    _write_overlay(overlay_compact, exit_depth=12)
    _write_overlay(overlay_spaced, exit_depth=12)
    # Intentionally different YAML formatting for the same merged structure.
    overlay_spaced.write_text(
        textwrap.dedent(
            """\
            method:
                name: fixed_exit
                exit_depth: 12
                fusion: equal

            adaptive:
                fixed_exit_depth: 12
                fusion_mode: equal
                early_depths: []
                full_depth: 12
            """
        ),
        encoding="utf-8",
    )

    compact_identity = adaptive_config_identity(BASE_CONFIG, overlay_path=overlay_compact)
    spaced_identity = adaptive_config_identity(BASE_CONFIG, overlay_path=overlay_spaced)

    assert compact_identity.overlay_sha256 != spaced_identity.overlay_sha256
    assert (
        compact_identity.effective_config_sha256
        == spaced_identity.effective_config_sha256
    )


def test_canonical_hash_ignores_key_order() -> None:
    left = {"adaptive": {"full_depth": 18, "fusion_mode": "equal"}, "method": {"name": "fixed_exit"}}
    right = {"method": {"name": "fixed_exit"}, "adaptive": {"fusion_mode": "equal", "full_depth": 18}}
    assert canonical_config_sha256(left) == canonical_config_sha256(right)


def test_deep_merge_matches_evaluator_semantics() -> None:
    base = load_yaml_mapping(BASE_CONFIG)
    overlay = load_yaml_mapping(OVERLAY_A)
    merged_once = deep_merge_config(base, overlay)
    merged_twice = deep_merge_config(deep_merge_config(base, {}), overlay)
    assert canonical_config_sha256(merged_once) == canonical_config_sha256(merged_twice)

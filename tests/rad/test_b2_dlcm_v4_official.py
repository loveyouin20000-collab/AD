"""C3A: official V4 training is deferred to C3B; config stub keeps training disabled."""

from __future__ import annotations

import json
from pathlib import Path


def test_official_config_stub_training_disabled() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = json.loads(
        (root / "configs/phase_b/b2_dlcm_uniform_relative_official_v4.json").read_text(
            encoding="utf-8"
        )
    )
    assert cfg["schema_version"] == "b2_dlcm_uniform_relative_official_v4"
    assert cfg["real_training_enabled"] is False
    assert cfg["official_training_enabled"] is False
    assert cfg["smoothmax_tau"] == 0.05
    assert cfg.get("gt_deployment_aggregation") == "uniform_relative_smooth_max"
    assert cfg.get("checkpoint_selection") == "constrained_worst_relative_regret"

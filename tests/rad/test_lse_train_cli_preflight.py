from __future__ import annotations

import pytest

import tools.train_lse as train_lse


def test_legacy_lse_dry_run_requires_b2_accepted_gate() -> None:
    with pytest.raises(SystemExit) as exc:
        train_lse.main(["--config", "configs/rad/lse.yaml", "--dry-run", "--device", "cpu"])
    assert exc.value.code == 2

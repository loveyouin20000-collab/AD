"""V5 beta grid tests."""

from __future__ import annotations

import pytest

from rad.phase_b import b2_dlcm_v5 as v5


def test_exact_101_grid() -> None:
    items = list(v5.iter_beta_grid())
    assert len(items) == 101
    assert items[0]["beta_index"] == 0
    assert items[-1]["beta_index"] == 100
    assert items[0]["beta_decimal"] == "0.00"
    assert items[-1]["beta_decimal"] == "1.00"
    assert items[37]["beta"] == 0.37
    assert items[37]["beta_decimal"] == "0.37"
    v5.validate_beta_grid()


def test_invalid_index() -> None:
    with pytest.raises(v5.B2DLCMV5Error) as exc:
        v5.beta_from_index(101)
    assert exc.value.code == "B2_DLCM_V5_BETA_GRID_INVALID"

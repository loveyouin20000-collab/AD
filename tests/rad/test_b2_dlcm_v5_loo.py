"""V5 LOO objective tests."""

from __future__ import annotations

from rad.phase_b import b2_dlcm_v5 as v5
from rad.phase_b import b2_dlcm_v5_calibration as calibration
from tests.rad.b2_dlcm_v5_fixtures import make_calibration_records


def test_loo_eight_folds_depth24() -> None:
    records = make_calibration_records()
    loo = calibration.leave_one_out_regrets(records, beta=0.5)
    assert loo["depth"] == 24
    assert len(loo["folds"]) == 8
    cats = [f["category"] for f in loo["folds"]]
    assert cats.count("bottle") == 4
    assert cats.count("carpet") == 4


def test_negative_regret_retained() -> None:
    records = make_calibration_records()
    # beta=1 uses dynamic weights which should often beat uniform on bottle fixtures
    loo = calibration.leave_one_out_regrets(records, beta=1.0)
    values = [float(f["relative_regret"]) for f in loo["folds"]]
    assert any(v < 0 for v in values)
    # no abs/clamp: values are raw floats
    assert all(v == v for v in values)


def test_kl_from_weights_matches_softmax_path() -> None:
    import torch
    import torch.nn.functional as F

    from rad.phase_b import b2_dlcm_v4 as v4

    p = F.softmax(torch.tensor([1.0, 0.5, 0.2]), dim=0)
    logits = torch.tensor([0.8, 0.3, 0.1])
    w = F.softmax(logits, dim=0)
    kl_w = v5.per_sample_allocation_kl_from_weights(p, w)
    kl_l = v4.per_sample_allocation_kl(p.unsqueeze(0), logits.unsqueeze(0))[0]
    assert torch.allclose(kl_w, kl_l, rtol=0, atol=1e-6)

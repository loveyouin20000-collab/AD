from __future__ import annotations

import torch

from rad.models.lse import LSE
from rad.trainers.lse_trainer import LSETrainer


def _synthetic_batch(b: int = 4, state_dim: int = 26):
    # Mix depths 12 and 18
    depth_id = torch.tensor([12, 18, 12, 18][:b], dtype=torch.long)
    return {
        "state": torch.randn(b, state_dim),
        "depth_id": depth_id,
        "target_gain": torch.rand(b) * 0.5,
        "target_sufficient": (torch.rand(b) > 0.5).float(),
    }


def test_one_optimizer_step_decreases_lse_loss():
    torch.manual_seed(0)
    state_dim = 26
    model = LSE(state_dim=state_dim, early_depths=(12, 18))
    trainer = LSETrainer(model=model, early_depths=(12, 18))
    opt = torch.optim.Adam(trainer.parameters(), lr=1e-2)
    batch = _synthetic_batch(state_dim=state_dim)

    with torch.no_grad():
        before = float(trainer.compute_loss(batch)["loss"])
    metrics = trainer.training_step(batch, opt)
    with torch.no_grad():
        after = float(trainer.compute_loss(batch)["loss"])

    assert metrics["loss"] > 0
    assert "nll" in metrics and "bce" in metrics
    assert after < before


def test_evaluate_reports_per_depth_metrics():
    torch.manual_seed(1)
    model = LSE(state_dim=8, early_depths=(12, 18))
    trainer = LSETrainer(model=model, early_depths=(12, 18), epsilon_gain=0.05)
    # Two batches covering both depths
    batches = [
        {
            "state": torch.randn(3, 8),
            "depth_id": torch.tensor([12, 12, 12]),
            "target_gain": torch.tensor([0.2, 0.0, 0.1]),
            "target_sufficient": torch.tensor([0.0, 1.0, 0.0]),
        },
        {
            "state": torch.randn(2, 8),
            "depth_id": torch.tensor([18, 18]),
            "target_gain": torch.tensor([0.0, 0.3]),
            "target_sufficient": torch.tensor([1.0, 0.0]),
        },
    ]
    report = trainer.evaluate(batches)
    for d in (12, 18):
        m = report[d]
        for key in ("mae", "rmse", "auroc", "brier", "ece", "nll"):
            assert key in m
            assert m[key] == m[key]  # not NaN
    assert "predictions" in report
    assert len(report["predictions"]) == 5

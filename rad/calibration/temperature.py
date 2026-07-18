from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return logits / float(temperature)


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    init: float = 1.0,
    max_iter: int = 50,
    lr: float = 0.05,
) -> float:
    """Fit a scalar temperature by minimizing BCEWithLogits on calibration logits.

    Optimizes log-temperature unconstrained for positivity.
    """
    if logits.shape != labels.shape:
        raise ValueError("logits/labels shape mismatch")
    log_t = torch.nn.Parameter(torch.tensor(float(init), dtype=logits.dtype).log())
    opt = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")

    logits_d = logits.detach()
    labels_d = labels.detach().to(dtype=logits.dtype)

    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        t = log_t.exp().clamp_min(1e-3)
        loss = F.binary_cross_entropy_with_logits(logits_d / t, labels_d)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().clamp_min(1e-3).detach().item())

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rad.models.lse import GainPrediction, LSE, heteroscedastic_gaussian_nll, lse_loss


class LSETrainer(nn.Module):
    """Train / evaluate LSE on frozen descriptor states and residual-gain targets."""

    def __init__(
        self,
        model: LSE,
        early_depths: Sequence[int] = (12, 18),
        epsilon_gain: float = 0.05,
        sufficiency_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.model = model
        self.early_depths = tuple(int(d) for d in early_depths)
        self.epsilon_gain = float(epsilon_gain)
        self.sufficiency_weight = float(sufficiency_weight)

    def compute_loss(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        pred = self.model(batch["state"], batch["depth_id"])
        nll = heteroscedastic_gaussian_nll(
            pred.mean, pred.log_variance, batch["target_gain"]
        ).mean()
        bce = F.binary_cross_entropy_with_logits(
            pred.sufficiency_logit,
            batch["target_sufficient"].to(dtype=pred.sufficiency_logit.dtype),
        )
        loss = nll + self.sufficiency_weight * bce
        return {"loss": loss, "nll": nll.detach(), "bce": bce.detach()}

    def training_step(
        self,
        batch: dict[str, Any],
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        self.train()
        optimizer.zero_grad(set_to_none=True)
        metrics = self.compute_loss(batch)
        metrics["loss"].backward()
        optimizer.step()
        return {k: float(v.detach()) for k, v in metrics.items()}

    @torch.no_grad()
    def predict_batch(self, batch: dict[str, Any]) -> GainPrediction:
        self.eval()
        return self.model(batch["state"], batch["depth_id"])

    @torch.no_grad()
    def evaluate(self, batches: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Per-depth metrics + flat prediction table.

        Metrics at each early depth: mae, rmse, auroc, brier, ece, nll.
        AUROC uses beneficial-depth labels ``gain > epsilon_gain`` scored by predicted mean.
        """
        rows: list[dict[str, Any]] = []
        for batch in batches:
            pred = self.predict_batch(batch)
            b = batch["state"].shape[0]
            sample_ids = batch.get("sample_id")
            for i in range(b):
                row: dict[str, Any] = {
                    "depth": int(batch["depth_id"][i].item()),
                    "target_gain": float(batch["target_gain"][i].item()),
                    "target_sufficient": float(batch["target_sufficient"][i].item()),
                    "pred_mean": float(pred.mean[i].item()),
                    "pred_log_variance": float(pred.log_variance[i].item()),
                    "pred_suf_logit": float(pred.sufficiency_logit[i].item()),
                    "pred_suf_prob": float(torch.sigmoid(pred.sufficiency_logit[i]).item()),
                }
                if sample_ids is not None:
                    row["sample_id"] = sample_ids[i]
                rows.append(row)

        report: dict[str, Any] = {"predictions": rows}
        for depth in self.early_depths:
            subset = [r for r in rows if r["depth"] == depth]
            report[depth] = _metrics_for_rows(subset, epsilon_gain=self.epsilon_gain)
        # Overall mean NLL for early stopping convenience
        if rows:
            nlls = [
                float(
                    heteroscedastic_gaussian_nll(
                        torch.tensor([r["pred_mean"]]),
                        torch.tensor([r["pred_log_variance"]]),
                        torch.tensor([r["target_gain"]]),
                    ).item()
                )
                for r in rows
            ]
            report["nll"] = float(sum(nlls) / len(nlls))
        else:
            report["nll"] = float("inf")
        return report


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, y_score))


def _expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    if probs.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1] if i < n_bins - 1 else probs <= bins[i + 1])
        if not np.any(mask):
            continue
        conf = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        ece += abs(conf - acc) * (float(mask.sum()) / float(probs.size))
    return float(ece)


def _metrics_for_rows(rows: list[dict[str, Any]], *, epsilon_gain: float) -> dict[str, float]:
    if not rows:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "auroc": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
            "nll": float("nan"),
            "n": 0,
        }
    tgt = np.array([r["target_gain"] for r in rows], dtype=np.float64)
    mean = np.array([r["pred_mean"] for r in rows], dtype=np.float64)
    logvar = np.array([r["pred_log_variance"] for r in rows], dtype=np.float64)
    suf_prob = np.array([r["pred_suf_prob"] for r in rows], dtype=np.float64)
    suf_tgt = np.array([r["target_sufficient"] for r in rows], dtype=np.float64)

    err = mean - tgt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    beneficial = (tgt > epsilon_gain).astype(np.float64)
    auroc = _safe_auroc(beneficial, mean)
    brier = float(np.mean((suf_prob - suf_tgt) ** 2))
    ece = _expected_calibration_error(suf_prob, suf_tgt)
    nll = float(
        heteroscedastic_gaussian_nll(
            torch.from_numpy(mean).float(),
            torch.from_numpy(logvar).float(),
            torch.from_numpy(tgt).float(),
        )
        .mean()
        .item()
    )
    return {
        "mae": mae,
        "rmse": rmse,
        "auroc": auroc,
        "brier": brier,
        "ece": ece,
        "nll": nll,
        "n": len(rows),
    }

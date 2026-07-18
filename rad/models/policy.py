from __future__ import annotations

from dataclasses import dataclass

import torch

from rad.models.lse import GainPrediction


@dataclass(frozen=True)
class ExitSignals:
    """Observable signals at an early-exit checkpoint."""

    map_uncertainty: float
    image_score: float
    stability: float


@dataclass(frozen=True)
class PolicyProfile:
    """Calibrated early-exit profile."""

    name: str
    gain_threshold: float
    kappa: float
    map_uncertainty_threshold: float
    image_confidence_margin: float
    stability_threshold: float
    require_map_uncertainty: bool = False
    require_image_confidence: bool = False
    require_stability: bool = False

    @classmethod
    def aggressive(cls, *, gain_threshold: float, kappa: float) -> PolicyProfile:
        return cls(
            name="aggressive",
            gain_threshold=gain_threshold,
            kappa=kappa,
            map_uncertainty_threshold=1.0,
            image_confidence_margin=0.0,
            stability_threshold=1.0,
            require_map_uncertainty=False,
            require_image_confidence=False,
            require_stability=False,
        )

    @classmethod
    def balanced(
        cls,
        *,
        gain_threshold: float,
        kappa: float,
        map_uncertainty_threshold: float,
        image_confidence_margin: float,
    ) -> PolicyProfile:
        return cls(
            name="balanced",
            gain_threshold=gain_threshold,
            kappa=kappa,
            map_uncertainty_threshold=map_uncertainty_threshold,
            image_confidence_margin=image_confidence_margin,
            stability_threshold=1.0,
            require_map_uncertainty=True,
            require_image_confidence=True,
            require_stability=False,
        )

    @classmethod
    def conservative(
        cls,
        *,
        gain_threshold: float,
        kappa: float,
        map_uncertainty_threshold: float,
        image_confidence_margin: float,
        stability_threshold: float,
    ) -> PolicyProfile:
        return cls(
            name="conservative",
            gain_threshold=gain_threshold,
            kappa=kappa,
            map_uncertainty_threshold=map_uncertainty_threshold,
            image_confidence_margin=image_confidence_margin,
            stability_threshold=stability_threshold,
            require_map_uncertainty=True,
            require_image_confidence=True,
            require_stability=True,
        )


def gain_ucb(prediction: GainPrediction, kappa: float) -> torch.Tensor:
    """Upper confidence bound on residual gain: mean + kappa * std."""
    std = torch.exp(0.5 * prediction.log_variance)
    return prediction.mean + float(kappa) * std


def _image_confident(image_score: float, margin: float) -> bool:
    """Symmetric confidence: far from 0.5 by at least margin."""
    return abs(float(image_score) - 0.5) >= float(margin)


def should_exit(
    prediction: GainPrediction,
    signals: ExitSignals,
    profile: PolicyProfile,
) -> bool:
    """Return True iff all enabled exit conditions for the profile hold."""
    ucb = float(gain_ucb(prediction, profile.kappa).reshape(-1)[0].item())
    if ucb > profile.gain_threshold:
        return False
    if profile.require_map_uncertainty and signals.map_uncertainty > profile.map_uncertainty_threshold:
        return False
    if profile.require_image_confidence and not _image_confident(
        signals.image_score, profile.image_confidence_margin
    ):
        return False
    if profile.require_stability and signals.stability > profile.stability_threshold:
        return False
    return True

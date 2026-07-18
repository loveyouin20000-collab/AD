"""Trainers for RAD-VisualAD staged modules."""

from rad.trainers.fusion_trainer import (
    FusionForwardResult,
    FusionLossWeights,
    FusionTrainer,
    compute_fusion_objective,
    pixel_average_precision,
)
from rad.trainers.joint_trainer import (
    GateResult,
    JointStepOutput,
    JointTrainer,
    NoRegressionThresholds,
    StagedCheckpointInfo,
    compute_cost_weight,
    evaluate_no_regression,
    soft_expected_depth_ratio,
    validate_staged_checkpoint_pair,
)
from rad.trainers.lse_trainer import LSEForwardResult, LSETrainer, compute_lse_objective

__all__ = [
    "FusionTrainer",
    "FusionForwardResult",
    "FusionLossWeights",
    "compute_fusion_objective",
    "pixel_average_precision",
    "LSETrainer",
    "LSEForwardResult",
    "compute_lse_objective",
    "JointTrainer",
    "JointStepOutput",
    "StagedCheckpointInfo",
    "NoRegressionThresholds",
    "GateResult",
    "compute_cost_weight",
    "soft_expected_depth_ratio",
    "evaluate_no_regression",
    "validate_staged_checkpoint_pair",
]

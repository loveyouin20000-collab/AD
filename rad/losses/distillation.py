from __future__ import annotations

import torch
import torch.nn.functional as F


def normalized_binary_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Binary entropy H(p)/log(2) in [0, 1]."""
    p = probs.clamp(eps, 1.0 - eps)
    ent = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
    return ent / torch.log(torch.tensor(2.0, device=probs.device, dtype=probs.dtype))


def confidence_weighted_distillation(
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> torch.Tensor:
    """Pixel KD weighted by 1 - normalized binary entropy of teacher probs.

    Args:
        student: logits [B, 1, H, W] (or [B, H, W])
        teacher: teacher logits, same shape
    Returns:
        scalar mean weighted L1 on probabilities (or logits-sigmoid space)
    """
    if student.ndim == 3:
        student = student.unsqueeze(1)
    if teacher.ndim == 3:
        teacher = teacher.unsqueeze(1)
    if student.shape != teacher.shape:
        raise ValueError(f"student/teacher shape mismatch: {student.shape} vs {teacher.shape}")

    with torch.no_grad():
        teacher_prob = torch.sigmoid(teacher)
        weight = 1.0 - normalized_binary_entropy(teacher_prob)

    student_prob = torch.sigmoid(student)
    # Detach teacher targets; weight focuses learning on confident teacher pixels
    diff = (student_prob - teacher_prob.detach()).abs()
    weighted = weight * diff
    denom = weight.sum().clamp_min(1e-8)
    return weighted.sum() / denom

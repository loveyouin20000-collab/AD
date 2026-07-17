from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class StageCache:
    """Cached ViT state for resumable staged execution.

    sequence: [tokens, batch, width]
    patch_tokens: depth -> [batch, patches, width]
    next_block: 1-based index of the next residual block to run
    """

    sequence: torch.Tensor
    next_block: int
    patch_tokens: dict[int, torch.Tensor]
    checkpoint_tokens: dict[int, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.next_block < 1:
            raise ValueError("next_block must be a positive 1-based block index")
        if self.sequence.ndim != 3:
            raise ValueError("sequence must have shape [tokens, batch, width]")
        for depth, tokens in self.patch_tokens.items():
            if tokens.ndim != 3:
                raise ValueError(
                    f"patch_tokens[{depth}] must have shape [batch, patches, width]"
                )
        for depth, tokens in self.checkpoint_tokens.items():
            if tokens.ndim != 3:
                raise ValueError(
                    f"checkpoint_tokens[{depth}] must have shape [batch, patches, width]"
                )

    def detach(self) -> StageCache:
        return StageCache(
            sequence=self.sequence.detach(),
            next_block=self.next_block,
            patch_tokens={k: v.detach() for k, v in self.patch_tokens.items()},
            checkpoint_tokens={k: v.detach() for k, v in self.checkpoint_tokens.items()},
        )


@dataclass(frozen=True)
class CheckpointOutput:
    """Tokens available at a candidate checkpoint depth."""

    depth: int
    patch_tokens: torch.Tensor
    anomaly_token: torch.Tensor
    normal_token: torch.Tensor
    class_token: torch.Tensor

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("depth must be a positive 1-based layer index")
        if self.patch_tokens.ndim != 3:
            raise ValueError("patch_tokens must have shape [batch, patches, width]")

    def detach(self) -> CheckpointOutput:
        return CheckpointOutput(
            depth=self.depth,
            patch_tokens=self.patch_tokens.detach(),
            anomaly_token=self.anomaly_token.detach(),
            normal_token=self.normal_token.detach(),
            class_token=self.class_token.detach(),
        )

"""SLM for car/sedan/SUV driving in GTA V."""
from __future__ import annotations

from typing import Optional

import torch

from models.base_model import ActionOutput, BaseSLM
from utils.registry import register


@register("models", "car")
class CarSLM(BaseSLM):
    """Mode-specific SLM for car/sedan/SUV driving."""

    def __init__(
        self,
        backbone: str = "custom_cnn",
        binary_dim: int = 9,
        continuous_dim: int = 2,
        hidden_dim: int = 256,
        frame_stack: int = 4,
        pretrained: bool = False,
        img_size: int = 224,
        **kwargs,
    ):
        super().__init__(
            mode="car",
            binary_dim=binary_dim,
            continuous_dim=continuous_dim,
            hidden_dim=hidden_dim,
            backbone=backbone,
            frame_stack=frame_stack,
            pretrained=pretrained,
            img_size=img_size,
            **kwargs,
        )

    def forward(
        self,
        frames: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> ActionOutput:
        features = self.encoder(frames)
        if context is not None:
            features = features + context  # simple residual merge
        return ActionOutput(
            binary_actions=self.binary_head(features),
            continuous_actions=self.continuous_head(features),
        )

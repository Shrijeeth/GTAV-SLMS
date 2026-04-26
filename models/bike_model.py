"""SLM for bicycle and motorcycle riding in GTA V."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.base_model import ActionOutput, BaseSLM
from utils.registry import register


def _simple_cnn(in_channels: int, hidden_dim: int) -> nn.Sequential:
    """Lightweight fallback CNN backbone."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, 5, stride=2, padding=2),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, hidden_dim),
        nn.ReLU(),
    )


@register("models", "bike")
class BikeSLM(BaseSLM):
    """Mode-specific SLM for bicycle and motorcycle riding."""

    def build_backbone(self, backbone: str, **kwargs) -> nn.Module:
        in_channels = kwargs.get("frame_stack", 4) * 3  # RGB * stack
        if backbone == "custom_cnn":
            return _simple_cnn(in_channels, self.hidden_dim)
        # TODO: integrate timm backbones (mobilevit, efficientvit, tinyvit)
        return _simple_cnn(in_channels, self.hidden_dim)

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

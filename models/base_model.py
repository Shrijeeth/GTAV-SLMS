"""Base class for all GTA V SLM models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from models.backbone_factory import create_backbone


@dataclass
class ActionOutput:
    """Unified action output from any mode model."""
    binary_actions: torch.Tensor     # (B, num_binary) logits
    continuous_actions: torch.Tensor  # (B, num_continuous) values


class BaseSLM(nn.Module, ABC):
    """Abstract base for all mode-specific SLMs.

    The backbone is created via :func:`models.backbone_factory.create_backbone`
    which supports ``custom_cnn``, ``mobilevit``, ``efficientvit``, and
    ``tinyvit``.  Subclasses only need to implement ``forward``.
    """

    def __init__(
        self,
        mode: str,
        binary_dim: int,
        continuous_dim: int,
        hidden_dim: int = 256,
        backbone: str = "custom_cnn",
        frame_stack: int = 4,
        channels_per_frame: int = 3,
        pretrained: bool = False,
        img_size: int = 224,
        **kwargs,
    ):
        super().__init__()
        self.mode = mode
        self.binary_dim = binary_dim
        self.continuous_dim = continuous_dim
        self.hidden_dim = hidden_dim
        self.backbone_name = backbone
        self.frame_stack = frame_stack
        self.channels_per_frame = channels_per_frame

        in_channels = frame_stack * channels_per_frame

        self.encoder = create_backbone(
            backbone_type=backbone,
            hidden_dim=hidden_dim,
            in_channels=in_channels,
            pretrained=pretrained,
            img_size=img_size,
        )

        self.binary_head = nn.Linear(hidden_dim, binary_dim)
        self.continuous_head = nn.Linear(hidden_dim, continuous_dim)

    @abstractmethod
    def forward(
        self,
        frames: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> ActionOutput:
        """Produce actions from stacked frames and optional context."""
        ...

    def predict(
        self,
        frames: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> dict:
        """Convenience wrapper returning numpy-friendly dict."""
        self.eval()
        with torch.no_grad():
            out = self.forward(frames, context)
        return {
            "binary": (out.binary_actions.sigmoid() > 0.5).cpu().numpy(),
            "continuous": out.continuous_actions.cpu().numpy(),
        }

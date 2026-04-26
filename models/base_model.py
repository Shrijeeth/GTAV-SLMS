"""Base class for all GTA V SLM models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ActionOutput:
    """Unified action output from any mode model."""
    binary_actions: torch.Tensor     # (B, num_binary) logits
    continuous_actions: torch.Tensor  # (B, num_continuous) values


class BaseSLM(nn.Module, ABC):
    """Abstract base for all mode-specific SLMs.

    Subclasses must implement ``build_backbone`` and ``forward``.
    """

    def __init__(self, mode: str, binary_dim: int, continuous_dim: int,
                 hidden_dim: int = 256, backbone: str = "custom_cnn",
                 **kwargs):
        super().__init__()
        self.mode = mode
        self.binary_dim = binary_dim
        self.continuous_dim = continuous_dim
        self.hidden_dim = hidden_dim
        self.backbone_name = backbone

        self.encoder = self.build_backbone(backbone, **kwargs)
        self.binary_head = nn.Linear(hidden_dim, binary_dim)
        self.continuous_head = nn.Linear(hidden_dim, continuous_dim)

    @abstractmethod
    def build_backbone(self, backbone: str, **kwargs) -> nn.Module:
        """Return a feature extractor that maps frames -> (B, hidden_dim)."""
        ...

    @abstractmethod
    def forward(self, frames: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> ActionOutput:
        """Produce actions from stacked frames and optional context."""
        ...

    def predict(self, frames: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> dict:
        """Convenience wrapper returning numpy-friendly dict."""
        self.eval()
        with torch.no_grad():
            out = self.forward(frames, context)
        return {
            "binary": (out.binary_actions.sigmoid() > 0.5).cpu().numpy(),
            "continuous": out.continuous_actions.cpu().numpy(),
        }

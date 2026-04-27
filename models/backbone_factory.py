"""Backbone factory — create CNN or timm-based vision backbones.

Supported backbone types:
    * ``custom_cnn``   – lightweight 3-layer CNN (default, no extra deps)
    * ``mobilevit``    – MobileViT-S via timm
    * ``efficientvit`` – EfficientViT-M5 via timm
    * ``tinyvit``      – TinyViT-21M-224 via timm
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

# ── timm model names (verified against timm >= 0.9) ──────────────────
TIMM_MODEL_MAP: dict[str, str] = {
    "mobilevit":    "mobilevit_s",
    "efficientvit": "efficientvit_m5.r224_in1k",
    "tinyvit":      "tiny_vit_21m_224.dist_in22k_ft_in1k",
}

SUPPORTED_BACKBONES: list[str] = ["custom_cnn"] + list(TIMM_MODEL_MAP)


# ── helpers ───────────────────────────────────────────────────────────
def get_supported_backbones() -> list[str]:
    """Return the list of supported backbone type strings."""
    return list(SUPPORTED_BACKBONES)


def get_timm_model_name(backbone_type: str) -> Optional[str]:
    """Map a short backbone name to the full timm identifier."""
    return TIMM_MODEL_MAP.get(backbone_type)


def _simple_cnn(in_channels: int, hidden_dim: int) -> nn.Sequential:
    """Lightweight 3-conv CNN with BatchNorm, ending in (B, hidden_dim)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, 5, stride=2, padding=2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, hidden_dim),
        nn.ReLU(),
    )


def _infer_timm_feature_dim(
    model: nn.Module, in_channels: int = 3, img_size: int = 224,
) -> int:
    """Run a dummy forward pass to discover the timm backbone's output dim."""
    device = next(model.parameters()).device
    dummy = torch.zeros(1, in_channels, img_size, img_size, device=device)
    with torch.no_grad():
        out = model(dummy)
    return out.shape[-1]


# ── timm wrapper ──────────────────────────────────────────────────────
class TimmBackboneWrapper(nn.Module):
    """Wrap a timm feature extractor + optional channel adapter + projection.

    Parameters
    ----------
    timm_model_name : str
        Name recognised by ``timm.create_model``.
    hidden_dim : int
        Desired output dimensionality.
    in_channels : int
        Number of input channels (e.g. 12 for 4-frame RGB stack).
    pretrained : bool
        Whether to load ImageNet pretrained weights.
    img_size : int
        Spatial resolution of input images.
    """

    def __init__(
        self,
        timm_model_name: str,
        hidden_dim: int,
        in_channels: int = 3,
        pretrained: bool = False,
        img_size: int = 224,
    ):
        super().__init__()
        import timm

        # Create timm model as feature extractor (no classification head)
        self.backbone = timm.create_model(
            timm_model_name,
            pretrained=pretrained,
            num_classes=0,         # remove classifier
            global_pool="avg",
            in_chans=in_channels,
        )

        feat_dim = _infer_timm_feature_dim(self.backbone, in_channels, img_size)

        # Project timm features → hidden_dim
        if feat_dim != hidden_dim:
            self.proj = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)           # (B, feat_dim)
        return self.proj(features)            # (B, hidden_dim)


# ── factory function ──────────────────────────────────────────────────
def create_backbone(
    backbone_type: str,
    hidden_dim: int = 256,
    in_channels: int = 12,
    pretrained: bool = False,
    img_size: int = 224,
) -> nn.Module:
    """Create and return a backbone module that maps (B, C, H, W) → (B, hidden_dim).

    Parameters
    ----------
    backbone_type : str
        One of :pydata:`SUPPORTED_BACKBONES`.
    hidden_dim : int
        Target feature dimensionality.
    in_channels : int
        Number of input image channels.
    pretrained : bool
        Use pretrained weights for timm backbones.
    img_size : int
        Input spatial size (only used by timm backbones).

    Returns
    -------
    nn.Module
        A module producing ``(B, hidden_dim)`` tensors.

    Raises
    ------
    ValueError
        If *backbone_type* is not in :pydata:`SUPPORTED_BACKBONES`.
    """
    if backbone_type == "custom_cnn":
        return _simple_cnn(in_channels, hidden_dim)

    timm_name = TIMM_MODEL_MAP.get(backbone_type)
    if timm_name is None:
        raise ValueError(
            f"Unknown backbone_type={backbone_type!r}. "
            f"Supported: {SUPPORTED_BACKBONES}"
        )

    return TimmBackboneWrapper(
        timm_model_name=timm_name,
        hidden_dim=hidden_dim,
        in_channels=in_channels,
        pretrained=pretrained,
        img_size=img_size,
    )

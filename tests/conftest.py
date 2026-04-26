"""Shared pytest fixtures for GTA V SLM tests."""
import sys
import os

import pytest
import torch

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


BACKBONE_TYPES = ["custom_cnn", "mobilevit", "efficientvit", "tinyvit"]
MODES = ["walking", "car", "bike", "plane", "helicopter"]

MODE_DEFAULTS = {
    "walking":    {"binary_dim": 12, "continuous_dim": 2, "hidden_dim": 256, "frame_stack": 4},
    "car":        {"binary_dim": 9,  "continuous_dim": 2, "hidden_dim": 256, "frame_stack": 4},
    "bike":       {"binary_dim": 9,  "continuous_dim": 2, "hidden_dim": 192, "frame_stack": 4},
    "plane":      {"binary_dim": 12, "continuous_dim": 2, "hidden_dim": 320, "frame_stack": 6},
    "helicopter": {"binary_dim": 12, "continuous_dim": 2, "hidden_dim": 320, "frame_stack": 6},
}


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture(params=BACKBONE_TYPES)
def backbone_type(request):
    return request.param


@pytest.fixture(params=MODES)
def mode(request):
    return request.param


@pytest.fixture
def all_backbone_types():
    return list(BACKBONE_TYPES)


@pytest.fixture
def all_modes():
    return list(MODES)


@pytest.fixture
def dummy_frames():
    """Factory: create random frame tensor (B, C, H, W)."""
    def _make(batch_size=2, frame_stack=4, img_size=224):
        return torch.randn(batch_size, frame_stack * 3, img_size, img_size)
    return _make


@pytest.fixture
def sample_input():
    """4-frame RGB stacked input at 224x224."""
    return torch.randn(2, 12, 224, 224)


@pytest.fixture
def sample_rgb_input():
    """Single RGB frame at 224x224 (for 1-frame tests)."""
    return torch.randn(2, 3, 224, 224)

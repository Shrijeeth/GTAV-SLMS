"""Tests for mode-specific models with various backbone types."""
import pytest
import torch

from models.base_model import ActionOutput, BaseSLM
from models.walking_model import WalkingSLM
from models.car_model import CarSLM
from models.bike_model import BikeSLM
from models.plane_model import PlaneSLM
from models.helicopter_model import HelicopterSLM

from tests.conftest import MODE_DEFAULTS

MODEL_MAP = {
    "walking":    WalkingSLM,
    "car":        CarSLM,
    "bike":       BikeSLM,
    "plane":      PlaneSLM,
    "helicopter": HelicopterSLM,
}


class TestModelInstantiation:
    """Verify models create with all backbone types."""

    @pytest.mark.parametrize("mode", ["walking", "car", "bike", "plane", "helicopter"])
    def test_custom_cnn_instantiation(self, mode):
        params = MODE_DEFAULTS[mode]
        model = MODEL_MAP[mode](backbone="custom_cnn", pretrained=False, **params)
        assert isinstance(model, BaseSLM)
        assert model.mode == mode
        assert model.backbone_name == "custom_cnn"

    @pytest.mark.parametrize("mode", ["walking", "car", "bike", "plane", "helicopter"])
    @pytest.mark.parametrize("btype", ["mobilevit", "efficientvit", "tinyvit"])
    def test_timm_instantiation(self, mode, btype):
        params = MODE_DEFAULTS[mode]
        model = MODEL_MAP[mode](backbone=btype, pretrained=False, **params)
        assert isinstance(model, BaseSLM)
        assert model.backbone_name == btype


class TestModelForwardPass:
    """Verify forward pass produces correct output shapes."""

    @pytest.mark.parametrize("mode", ["walking", "car", "bike", "plane", "helicopter"])
    def test_forward_custom_cnn(self, mode):
        params = MODE_DEFAULTS[mode]
        model = MODEL_MAP[mode](backbone="custom_cnn", pretrained=False, **params)
        model.eval()
        x = torch.randn(2, params["frame_stack"] * 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert isinstance(out, ActionOutput)
        assert out.binary_actions.shape == (2, params["binary_dim"])
        assert out.continuous_actions.shape == (2, params["continuous_dim"])

    @pytest.mark.parametrize("mode", ["walking", "car", "bike", "plane", "helicopter"])
    @pytest.mark.parametrize("btype", ["mobilevit", "efficientvit", "tinyvit"])
    def test_forward_timm(self, mode, btype):
        params = MODE_DEFAULTS[mode]
        model = MODEL_MAP[mode](backbone=btype, pretrained=False, **params)
        model.eval()
        x = torch.randn(2, params["frame_stack"] * 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert isinstance(out, ActionOutput)
        assert out.binary_actions.shape == (2, params["binary_dim"])
        assert out.continuous_actions.shape == (2, params["continuous_dim"])


class TestBackwardCompatibility:
    """Default backbone should be custom_cnn — no breaking changes."""

    @pytest.mark.parametrize("mode", ["walking", "car", "bike", "plane", "helicopter"])
    def test_default_backbone_is_custom_cnn(self, mode):
        model = MODEL_MAP[mode]()
        assert model.backbone_name == "custom_cnn"

    @pytest.mark.parametrize("mode", ["walking", "car", "bike", "plane", "helicopter"])
    def test_predict_returns_dict(self, mode):
        params = MODE_DEFAULTS[mode]
        model = MODEL_MAP[mode](backbone="custom_cnn", pretrained=False, **params)
        x = torch.randn(1, params["frame_stack"] * 3, 224, 224)
        result = model.predict(x)
        assert isinstance(result, dict)
        assert "binary" in result
        assert "continuous" in result

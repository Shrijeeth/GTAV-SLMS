"""Tests for registry with backbone_type support."""
import pytest
import torch

import models  # noqa: F401 — trigger @register decorators
from utils.registry import get, list_registered, create_model
from models.base_model import BaseSLM, ActionOutput

from tests.conftest import MODE_DEFAULTS


class TestRegistryDiscovery:
    """Verify all models are registered."""

    def test_all_modes_registered(self):
        registered = list_registered("models")
        for mode in ["walking", "car", "bike", "plane", "helicopter"]:
            assert mode in registered, f"{mode} not registered"

    def test_get_model_class(self):
        cls = get("models", "walking")
        assert issubclass(cls, BaseSLM)

    def test_get_nonexistent_raises(self):
        with pytest.raises(KeyError):
            get("models", "spaceship")


class TestCreateModel:
    """Test registry.create_model with backbone_type."""

    @pytest.mark.parametrize("mode", ["walking", "car", "bike", "plane", "helicopter"])
    def test_create_with_custom_cnn(self, mode):
        params = MODE_DEFAULTS[mode]
        model = create_model(mode, backbone_type="custom_cnn", **params)
        assert isinstance(model, BaseSLM)
        assert model.backbone_name == "custom_cnn"

    @pytest.mark.parametrize("mode", ["walking", "car", "bike", "plane", "helicopter"])
    @pytest.mark.parametrize("btype", ["mobilevit", "efficientvit", "tinyvit"])
    def test_create_with_timm_backbone(self, mode, btype):
        params = MODE_DEFAULTS[mode]
        model = create_model(mode, backbone_type=btype, pretrained=False, **params)
        assert isinstance(model, BaseSLM)
        assert model.backbone_name == btype

    @pytest.mark.parametrize("mode", ["walking", "car", "bike", "plane", "helicopter"])
    def test_forward_pass(self, mode):
        params = MODE_DEFAULTS[mode]
        model = create_model(mode, backbone_type="custom_cnn", **params)
        model.eval()
        x = torch.randn(1, params["frame_stack"] * 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert isinstance(out, ActionOutput)

    def test_default_backbone_type(self):
        params = MODE_DEFAULTS["walking"]
        model = create_model("walking", **params)
        assert model.backbone_name == "custom_cnn"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            create_model("flying_saucer")

    def test_different_instances(self):
        m1 = create_model("walking")
        m2 = create_model("walking")
        assert m1 is not m2

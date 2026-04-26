"""Tests for BackboneFactory and timm backbone integration."""
import pytest
import torch
import torch.nn as nn

from models.backbone_factory import (
    create_backbone,
    get_supported_backbones,
    get_timm_model_name,
    SUPPORTED_BACKBONES,
    TIMM_MODEL_MAP,
    TimmBackboneWrapper,
    _simple_cnn,
)


class TestSupportedBackbones:
    """Verify backbone listing and metadata."""

    def test_supported_list_complete(self):
        backbones = get_supported_backbones()
        assert "custom_cnn" in backbones
        assert "mobilevit" in backbones
        assert "efficientvit" in backbones
        assert "tinyvit" in backbones
        assert len(backbones) == 4

    def test_timm_model_names_present(self):
        assert get_timm_model_name("custom_cnn") is None
        for name in ["mobilevit", "efficientvit", "tinyvit"]:
            assert get_timm_model_name(name) is not None

    def test_timm_map_keys_match_supported(self):
        timm_types = set(TIMM_MODEL_MAP.keys())
        expected = set(SUPPORTED_BACKBONES) - {"custom_cnn"}
        assert timm_types == expected


class TestCustomCNNBackbone:
    """Test the lightweight custom CNN backbone."""

    def test_create(self):
        backbone = create_backbone("custom_cnn", hidden_dim=256, in_channels=12)
        assert isinstance(backbone, nn.Sequential)

    def test_output_shape(self):
        backbone = create_backbone("custom_cnn", hidden_dim=256, in_channels=12)
        x = torch.randn(2, 12, 224, 224)
        out = backbone(x)
        assert out.shape == (2, 256)

    @pytest.mark.parametrize("dim", [128, 192, 256, 320, 512])
    def test_different_hidden_dims(self, dim):
        backbone = create_backbone("custom_cnn", hidden_dim=dim, in_channels=12)
        out = backbone(torch.randn(1, 12, 224, 224))
        assert out.shape == (1, dim)

    @pytest.mark.parametrize("size", [64, 128, 224])
    def test_different_spatial_sizes(self, size):
        backbone = create_backbone("custom_cnn", hidden_dim=256, in_channels=12)
        out = backbone(torch.randn(1, 12, size, size))
        assert out.shape == (1, 256)


class TestTimmBackbones:
    """Test timm-based backbones."""

    @pytest.mark.parametrize("btype", ["mobilevit", "efficientvit", "tinyvit"])
    def test_create_timm_backbone(self, btype):
        backbone = create_backbone(
            btype, hidden_dim=256, in_channels=12, pretrained=False
        )
        assert isinstance(backbone, TimmBackboneWrapper)

    @pytest.mark.parametrize("btype", ["mobilevit", "efficientvit", "tinyvit"])
    def test_output_shape(self, btype):
        backbone = create_backbone(
            btype, hidden_dim=256, in_channels=12, pretrained=False, img_size=224
        )
        out = backbone(torch.randn(2, 12, 224, 224))
        assert out.shape == (2, 256)

    @pytest.mark.parametrize("btype", ["mobilevit", "efficientvit", "tinyvit"])
    @pytest.mark.parametrize("dim", [192, 256, 320])
    def test_different_hidden_dims(self, btype, dim):
        backbone = create_backbone(
            btype, hidden_dim=dim, in_channels=12, pretrained=False
        )
        out = backbone(torch.randn(1, 12, 224, 224))
        assert out.shape == (1, dim)

    @pytest.mark.parametrize("btype", ["mobilevit", "efficientvit", "tinyvit"])
    def test_has_projection_layer(self, btype):
        backbone = create_backbone(
            btype, hidden_dim=128, in_channels=3, pretrained=False
        )
        assert hasattr(backbone, "proj")
        assert hasattr(backbone, "backbone")


class TestBackboneFactoryErrors:
    """Error handling in backbone factory."""

    def test_invalid_backbone_raises(self):
        with pytest.raises(ValueError, match="Unknown backbone_type"):
            create_backbone("nonexistent", hidden_dim=256)

    def test_error_lists_supported(self):
        with pytest.raises(ValueError, match="Supported"):
            create_backbone("resnet50", hidden_dim=256)

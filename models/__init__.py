"""GTA V SLM models — import submodules to trigger @register decorators."""
from models.backbone_factory import (  # noqa: F401
    create_backbone,
    get_supported_backbones,
    get_timm_model_name,
    SUPPORTED_BACKBONES,
    TIMM_MODEL_MAP,
    TimmBackboneWrapper,
)
from models.base_model import ActionOutput, BaseSLM  # noqa: F401

# Auto-register all mode models
from models import (  # noqa: F401
    walking_model,
    car_model,
    bike_model,
    plane_model,
    helicopter_model,
)

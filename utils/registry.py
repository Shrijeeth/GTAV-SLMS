"""Dynamic registry for models, modes, and components."""
from __future__ import annotations

import importlib
from typing import Any, Callable, Dict

_REGISTRIES: Dict[str, Dict[str, Any]] = {}


def register(namespace: str, name: str) -> Callable:
    """Decorator to register a class or function under *namespace*/*name*."""
    def decorator(obj: Any) -> Any:
        _REGISTRIES.setdefault(namespace, {})[name] = obj
        return obj
    return decorator


def get(namespace: str, name: str) -> Any:
    """Retrieve a registered object."""
    try:
        return _REGISTRIES[namespace][name]
    except KeyError:
        available = list(_REGISTRIES.get(namespace, {}).keys())
        raise KeyError(
            f"{name!r} not found in registry {namespace!r}. "
            f"Available: {available}"
        )


def list_registered(namespace: str) -> list[str]:
    """List all names in a namespace."""
    return list(_REGISTRIES.get(namespace, {}).keys())


def create_model(
    mode: str,
    backbone_type: str = "custom_cnn",
    **kwargs,
) -> Any:
    """Dynamically create a mode-specific model with the given backbone.

    Parameters
    ----------
    mode : str
        Game mode (walking, car, bike, plane, helicopter).
    backbone_type : str
        Backbone to use (custom_cnn, mobilevit, efficientvit, tinyvit).
    **kwargs
        Forwarded to the model constructor.

    Returns
    -------
    BaseSLM
        Instantiated mode-specific model.
    """
    _MODEL_MAP = {
        "walking":    "models.walking_model.WalkingSLM",
        "car":        "models.car_model.CarSLM",
        "bike":       "models.bike_model.BikeSLM",
        "plane":      "models.plane_model.PlaneSLM",
        "helicopter": "models.helicopter_model.HelicopterSLM",
    }
    if mode not in _MODEL_MAP:
        raise ValueError(
            f"Unknown mode: {mode!r}. Supported: {list(_MODEL_MAP.keys())}"
        )
    module_path, class_name = _MODEL_MAP[mode].rsplit(".", 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    return model_class(backbone=backbone_type, **kwargs)

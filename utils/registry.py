"""Dynamic registry for models, modes, and components."""
from __future__ import annotations
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

"""Translate between model action vectors and keyboard/mouse events."""
from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ActionSpec:
    """Specification for a single action dimension."""
    name: str
    type: str            # "binary" or "continuous"
    range: tuple | None = None
    description: str = ""


@dataclass
class ModeActionSpace:
    """Full action space for a game mode."""
    mode: str
    keys: List[ActionSpec] = field(default_factory=list)
    mouse: List[ActionSpec] = field(default_factory=list)

    @property
    def binary_dim(self) -> int:
        return len([a for a in self.keys + self.mouse if a.type == "binary"])

    @property
    def continuous_dim(self) -> int:
        return len([a for a in self.keys + self.mouse if a.type == "continuous"])

    @property
    def total_dim(self) -> int:
        return self.binary_dim + self.continuous_dim


def load_action_space(config_path: str | Path) -> ModeActionSpace:
    """Load an action space from a mode YAML config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    mode = cfg["mode"]
    keys = [
        ActionSpec(
            name=k["name"],
            type=k["type"],
            description=k.get("description", ""),
        )
        for k in cfg["action_space"].get("keys", [])
    ]
    mouse = []
    for name, spec in cfg["action_space"].get("mouse", {}).items():
        mouse.append(ActionSpec(
            name=name,
            type=spec["type"],
            range=tuple(spec["range"]) if "range" in spec else None,
            description=spec.get("description", ""),
        ))
    return ModeActionSpace(mode=mode, keys=keys, mouse=mouse)

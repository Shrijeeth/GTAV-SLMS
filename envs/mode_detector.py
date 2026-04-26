"""Detect current GTA V activity mode from screen / HUD analysis."""
from __future__ import annotations

from enum import Enum
from typing import Optional
import numpy as np


class GameMode(str, Enum):
    WALKING = "walking"
    CAR = "car"
    BIKE = "bike"
    PLANE = "plane"
    HELICOPTER = "helicopter"
    UNKNOWN = "unknown"


class ModeDetector:
    """Detect game mode from a captured frame.

    Current implementation: placeholder returning UNKNOWN.
    Future: HUD-region classifier or small CNN trained on labelled HUD crops.
    """

    def __init__(self) -> None:
        self._classifier = None  # TODO: load lightweight classifier

    def detect(self, frame: np.ndarray) -> GameMode:
        """Return the detected GameMode for the given frame."""
        # Placeholder — always returns UNKNOWN until classifier is trained
        return GameMode.UNKNOWN

    def detect_with_confidence(self, frame: np.ndarray) -> tuple[GameMode, float]:
        """Return (mode, confidence) pair."""
        mode = self.detect(frame)
        confidence = 0.0 if mode == GameMode.UNKNOWN else 1.0
        return mode, confidence

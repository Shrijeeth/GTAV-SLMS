"""Visualization helpers for frames, actions, and training metrics."""
from __future__ import annotations

import numpy as np


def overlay_actions_on_frame(
    frame: np.ndarray,
    actions: dict,
    position: tuple[int, int] = (10, 30),
) -> np.ndarray:
    """Draw action labels on a frame (requires cv2)."""
    import cv2
    frame = frame.copy()
    y = position[1]
    for key, val in actions.items():
        text = f"{key}: {val}"
        cv2.putText(frame, text, (position[0], y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y += 20
    return frame

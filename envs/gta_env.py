"""GTA V environment wrapper — ties screen capture to action execution."""
from __future__ import annotations

from typing import Optional
import numpy as np


class GTAEnvironment:
    """Lightweight wrapper around screen capture + input execution.

    Not a full Gym env (no reward signal) — used for inference and
    data-collection orchestration.
    """

    def __init__(self, capture_backend: str = "mss",
                 frame_width: int = 320, frame_height: int = 180):
        self.capture_backend = capture_backend
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._capturer = None

    def start(self) -> None:
        """Initialise screen capturer."""
        if self.capture_backend == "mss":
            import mss
            self._capturer = mss.mss()
        # TODO: add dxcam backend

    def grab_frame(self) -> np.ndarray:
        """Capture a single frame, resize, and return as (H, W, 3) uint8."""
        import cv2
        if self._capturer is None:
            self.start()
        monitor = self._capturer.monitors[1]
        img = np.array(self._capturer.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = cv2.resize(img, (self.frame_width, self.frame_height))
        return img

    def stop(self) -> None:
        if self._capturer is not None:
            self._capturer.close()
            self._capturer = None

"""Real-time inference runner for GTA V SLM."""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

from envs.gta_env import GTAEnvironment
from envs.mode_detector import GameMode, ModeDetector
from utils.registry import get as registry_get


class InferenceRunner:
    """Run a trained SLM in real-time against GTA V."""

    def __init__(self, config_path: str = "configs/inference.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        inf = self.cfg["inference"]
        self.target_fps = inf["target_fps"]
        self.auto_detect = inf["auto_detect_mode"]
        self.default_mode = inf["default_mode"]
        self.device = inf["device"]

        self.env = GTAEnvironment()
        self.detector = ModeDetector()
        self.models: dict[str, torch.nn.Module] = {}
        self.frame_buffer: deque = deque(maxlen=6)

    def load_model(self, mode: str, checkpoint: str | Path) -> None:
        """Load a mode-specific model from checkpoint."""
        # Dynamically import to trigger @register decorators
        import models.walking_model  # noqa
        import models.car_model      # noqa
        import models.bike_model     # noqa
        import models.plane_model    # noqa
        import models.helicopter_model  # noqa

        model_cls = registry_get("models", mode)
        # TODO: read dims from mode config
        model = model_cls(mode=mode, binary_dim=10, continuous_dim=4,
                          hidden_dim=256)
        model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        model.to(self.device).eval()
        self.models[mode] = model

    def _get_stacked_frames(self, n: int = 4) -> torch.Tensor:
        while len(self.frame_buffer) < n:
            self.frame_buffer.append(self.env.grab_frame())
        frames = list(self.frame_buffer)[-n:]
        stacked = np.concatenate(frames, axis=-1)  # (H, W, n*3)
        tensor = torch.from_numpy(
            np.transpose(stacked, (2, 0, 1)).astype(np.float32) / 255.0
        ).unsqueeze(0).to(self.device)
        return tensor

    def step(self) -> dict:
        """Capture frame, detect mode, run model, return actions."""
        frame = self.env.grab_frame()
        self.frame_buffer.append(frame)

        if self.auto_detect:
            mode = self.detector.detect(frame)
            if mode == GameMode.UNKNOWN:
                mode = GameMode(self.default_mode)
        else:
            mode = GameMode(self.default_mode)

        model = self.models.get(mode.value)
        if model is None:
            return {"mode": mode.value, "actions": None, "error": "no model loaded"}

        frames_t = self._get_stacked_frames()
        actions = model.predict(frames_t)
        return {"mode": mode.value, "actions": actions}

    def run(self) -> None:
        """Main inference loop."""
        self.env.start()
        interval = 1.0 / self.target_fps
        print(f"Starting inference at {self.target_fps} FPS (Ctrl-C to stop)")
        try:
            while True:
                t0 = time.time()
                result = self.step()
                # TODO: feed result["actions"] to ActionExecutor
                elapsed = time.time() - t0
                if elapsed < interval:
                    time.sleep(interval - elapsed)
        except KeyboardInterrupt:
            print("Inference stopped.")
        finally:
            self.env.stop()

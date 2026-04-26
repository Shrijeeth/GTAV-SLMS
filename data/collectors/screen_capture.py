"""Capture GTA V screen frames at configurable FPS."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np


def capture_loop(
    output_dir: str | Path,
    fps: int = 15,
    duration_s: Optional[float] = None,
    backend: str = "mss",
    frame_width: int = 320,
    frame_height: int = 180,
) -> None:
    """Capture frames in a loop and save as .npy files.

    Parameters
    ----------
    output_dir : path to store frames
    fps : target capture rate
    duration_s : stop after this many seconds (None = until Ctrl-C)
    """
    from envs.gta_env import GTAEnvironment

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = GTAEnvironment(capture_backend=backend,
                         frame_width=frame_width,
                         frame_height=frame_height)
    env.start()

    interval = 1.0 / fps
    idx = 0
    t0 = time.time()
    try:
        while True:
            ts = time.time()
            frame = env.grab_frame()
            np.save(out / f"frame_{idx:08d}.npy", frame)
            idx += 1
            elapsed = time.time() - ts
            if elapsed < interval:
                time.sleep(interval - elapsed)
            if duration_s and (time.time() - t0) >= duration_s:
                break
    except KeyboardInterrupt:
        pass
    finally:
        env.stop()
    print(f"Captured {idx} frames -> {out}")

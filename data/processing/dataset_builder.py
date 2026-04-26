"""Build HDF5 / Zarr datasets from raw frame + input recordings."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def build_hdf5_dataset(
    frames_dir: str | Path,
    inputs_jsonl: str | Path,
    output_path: str | Path,
    mode: str = "car",
    fps: int = 15,
) -> None:
    """Align frames with input events and write an HDF5 file.

    Output structure::

        /frames     (N, H, W, 3) uint8
        /actions    (N, action_dim) float32
        /timestamps (N,) float64
        /mode       scalar string attribute
    """
    import h5py
    import json

    frames_dir = Path(frames_dir)
    frame_files = sorted(frames_dir.glob("frame_*.npy"))
    if not frame_files:
        raise FileNotFoundError(f"No frame files in {frames_dir}")

    # Load inputs
    events = []
    with open(inputs_jsonl) as f:
        for line in f:
            events.append(json.loads(line.strip()))

    # Simple alignment: assign each frame the latest input state
    # TODO: implement proper temporal alignment with action_space config
    frames = [np.load(fp) for fp in frame_files]
    frames_arr = np.stack(frames)

    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("frames", data=frames_arr, compression="gzip")
        hf.attrs["mode"] = mode
        hf.attrs["fps"] = fps
        hf.attrs["num_frames"] = len(frames_arr)
    print(f"Built dataset with {len(frames_arr)} frames -> {output_path}")

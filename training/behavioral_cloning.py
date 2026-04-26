"""Behavioral cloning dataset and training utilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BCDataset(Dataset):
    """Dataset for behavioral cloning from HDF5 files."""

    def __init__(self, hdf5_path: str | Path, frame_stack: int = 4,
                 transform=None):
        import h5py
        self.hf = h5py.File(hdf5_path, "r")
        self.frames = self.hf["frames"]
        self.frame_stack = frame_stack
        self.transform = transform
        # actions dataset will be added when dataset_builder writes them
        self._has_actions = "actions" in self.hf

    def __len__(self) -> int:
        return max(len(self.frames) - self.frame_stack, 0)

    def __getitem__(self, idx: int) -> dict:
        # Stack last N frames -> (N*3, H, W)
        stack = []
        for i in range(self.frame_stack):
            f = self.frames[idx + i]
            if self.transform:
                f = self.transform(f)
            stack.append(f)
        frames = np.concatenate(stack, axis=-1)  # (H, W, N*3)
        frames = np.transpose(frames, (2, 0, 1)).astype(np.float32) / 255.0

        sample = {"frames": torch.from_numpy(frames)}
        if self._has_actions:
            act = self.hf["actions"][idx + self.frame_stack - 1]
            sample["binary_actions"] = torch.from_numpy(act[:10].astype(np.float32))
            sample["continuous_actions"] = torch.from_numpy(act[10:].astype(np.float32))
        return sample

    def __del__(self):
        if hasattr(self, "hf"):
            self.hf.close()

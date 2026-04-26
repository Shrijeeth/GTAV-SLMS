"""Generic trainer loop for GTA V SLMs."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """Minimal training loop with mixed-precision and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        epochs: int = 100,
        device: str = "cuda",
        mixed_precision: bool = True,
        checkpoint_dir: str = "checkpoints",
        log_backend: str = "wandb",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scaler = torch.amp.GradScaler(enabled=mixed_precision)
        self.mixed_precision = mixed_precision
        self.best_val_loss = float("inf")

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(self.train_loader):
            frames = batch["frames"].to(self.device)
            binary_targets = batch["binary_actions"].to(self.device)
            continuous_targets = batch["continuous_actions"].to(self.device)

            with torch.amp.autocast("cuda", enabled=self.mixed_precision):
                out = self.model(frames)
                loss_bin = nn.functional.binary_cross_entropy_with_logits(
                    out.binary_actions, binary_targets
                )
                loss_cont = nn.functional.mse_loss(
                    out.continuous_actions, continuous_targets
                )
                loss = loss_bin + loss_cont

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()

        return total_loss / max(len(self.train_loader), 1)

    def validate(self) -> float:
        if self.val_loader is None:
            return 0.0
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                frames = batch["frames"].to(self.device)
                binary_targets = batch["binary_actions"].to(self.device)
                continuous_targets = batch["continuous_actions"].to(self.device)
                with torch.amp.autocast("cuda", enabled=self.mixed_precision):
                    out = self.model(frames)
                    loss_bin = nn.functional.binary_cross_entropy_with_logits(
                        out.binary_actions, binary_targets
                    )
                    loss_cont = nn.functional.mse_loss(
                        out.continuous_actions, continuous_targets
                    )
                    total_loss += (loss_bin + loss_cont).item()
        return total_loss / max(len(self.val_loader), 1)

    def fit(self) -> None:
        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate()
            print(f"Epoch {epoch+1}/{self.epochs}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                path = self.checkpoint_dir / f"{self.model.mode}_best.pt"
                torch.save(self.model.state_dict(), path)
                print(f"  -> saved best checkpoint: {path}")

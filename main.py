"""GTA V SLM — CLI entry point."""
from __future__ import annotations

import click


@click.group()
def cli():
    """GTA V Small Language Models for gameplay automation."""
    pass


@cli.command()
@click.option("--mode", "-m", default="car", help="Game mode to record.")
@click.option("--fps", default=15, help="Target capture FPS.")
@click.option("--output", "-o", required=True, help="Output directory.")
@click.option("--duration", "-d", default=None, type=float,
              help="Recording duration in seconds (default: until Ctrl-C).")
def collect(mode: str, fps: int, output: str, duration: float | None):
    """Collect gameplay data (screen + inputs)."""
    from data.collectors.screen_capture import capture_loop
    click.echo(f"Recording mode={mode} at {fps} FPS -> {output}")
    capture_loop(output_dir=output, fps=fps, duration_s=duration)


@cli.command()
@click.option("--mode", "-m", required=True, help="Game mode to train.")
@click.option("--config", "-c", default="configs/training.yaml",
              help="Training config path.")
@click.option("--data", "-d", required=True, help="Path to HDF5 dataset.")
def train(mode: str, config: str, data: str):
    """Train a mode-specific SLM via behavioral cloning."""
    click.echo(f"Training mode={mode} with config={config} data={data}")
    # TODO: wire up Trainer + BCDataset
    click.echo("Training pipeline not yet fully implemented.")


@cli.command()
@click.option("--mode", "-m", default=None,
              help="Game mode (default: auto-detect).")
@click.option("--checkpoint", "-ckpt", default=None,
              help="Model checkpoint path.")
@click.option("--auto-detect", is_flag=True, default=False,
              help="Auto-detect game mode from HUD.")
def infer(mode: str | None, checkpoint: str | None, auto_detect: bool):
    """Run real-time inference (AI plays GTA V)."""
    from inference.runner import InferenceRunner
    runner = InferenceRunner()
    if mode and checkpoint:
        runner.load_model(mode, checkpoint)
    elif auto_detect:
        click.echo("Auto-detect mode enabled.")
    else:
        click.echo("Provide --mode + --checkpoint or --auto-detect")
        return
    runner.run()


if __name__ == "__main__":
    cli()

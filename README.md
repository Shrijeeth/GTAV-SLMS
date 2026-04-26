# 🎮 GTA V - Small Language Models (SLMs) for Gameplay Automation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-red" />
  <img src="https://img.shields.io/badge/License-Apache%202.0-green" />
</p>

## Overview

This project builds **Small Language / Vision Models (SLMs)** that learn to play GTA V by watching human gameplay. Each activity mode in the game (walking, driving a car, riding a bike/motorcycle, flying a plane, piloting a helicopter) gets its own specialised lightweight model with a tailored control schema.

The pipeline:
1. **Collect** — capture screen frames + keyboard/mouse inputs while you play.
2. **Train** — behavioural-cloning SLMs per activity mode.
3. **Infer** — run the model in real-time to control the game.

## Architecture

```
┌─────────────┐   frames    ┌────────────────┐   actions   ┌──────────────┐
│  GTA V Game  │ ──────────► │  Mode Detector  │ ──────────► │  Mode Router  │
│  (Screen)    │             │  (HUD Analysis) │             │              │
└─────────────┘             └────────────────┘             └──────┬───────┘
                                                                  │
                     ┌────────────────────────────────────────────┤
                     ▼              ▼              ▼              ▼
              ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
              │  Walking  │  │   Car    │  │  Bike /  │  │  Plane / │
              │   Model   │  │  Model   │  │ Motor.   │  │  Heli    │
              └──────────┘  └──────────┘  └──────────┘  └──────────┘
                     │              │              │              │
                     └──────────────┴──────────────┴──────────────┘
                                        │
                                        ▼
                              Keyboard / Mouse Output
```

## Supported Activity Modes

| Mode | Model Key | Primary Controls |
|------|-----------|------------------|
| Walking / On-Foot | `walking` | WASD, Space, Shift, Mouse look |
| Car Driving | `car` | WASD, Space (brake), Mouse steer |
| Bike / Motorcycle | `bike` | WASD, Space, Shift (lean), Mouse |
| Plane Flying | `plane` | WASD (yaw/pitch), Num 4/6 (roll), Mouse |
| Helicopter Piloting | `helicopter` | WASD (dir.), Space/Ctrl (alt.), Mouse |

## Project Structure

```
GTAV-SLMS/
├── configs/             # YAML configs per mode & training
│   ├── modes/
│   │   ├── walking.yaml
│   │   ├── car.yaml
│   │   ├── bike.yaml
│   │   ├── plane.yaml
│   │   └── helicopter.yaml
│   ├── training.yaml
│   └── inference.yaml
├── data/                # Data collection scripts & storage
│   ├── collectors/
│   │   ├── screen_capture.py
│   │   └── input_logger.py
│   └── processing/
│       └── dataset_builder.py
├── envs/                # Environment wrappers & mode detection
│   ├── gta_env.py
│   └── mode_detector.py
├── models/              # Model architectures per mode
│   ├── base_model.py
│   ├── walking_model.py
│   ├── car_model.py
│   ├── bike_model.py
│   ├── plane_model.py
│   └── helicopter_model.py
├── training/            # Training loops & experiment scripts
│   ├── trainer.py
│   └── behavioral_cloning.py
├── inference/           # Real-time inference & action execution
│   ├── runner.py
│   └── action_executor.py
├── utils/               # Shared utilities
│   ├── registry.py
│   ├── input_mapping.py
│   └── visualization.py
├── main.py              # CLI entry point
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Quick Start

```bash
# Clone
git clone https://github.com/Shrijeeth/GTAV-SLMS.git
cd GTAV-SLMS

# Install
pip install -e .

# Collect data (play the game while recording)
python main.py collect --mode car --fps 15 --output data/raw/car_session1

# Train a model
python main.py train --mode car --config configs/training.yaml

# Run inference (AI plays)
python main.py infer --mode car --checkpoint checkpoints/car_best.pt

# Auto-detect mode and run
python main.py infer --auto-detect
```

## Configuration

Each mode has its own YAML config defining:
- **action_space**: keys, mouse axes, value ranges
- **observation_space**: frame size, stacking, extra HUD crops
- **model**: backbone, hidden dims, heads

See `configs/modes/` for per-mode configs and `configs/training.yaml` for hyper-parameters.

## Models

Backbone options (configured per mode):
- **MobileViT-XXS** — good speed/accuracy for driving
- **EfficientViT-B0** — balanced efficiency
- **TinyViT** — smallest footprint for walking
- **Custom CNN** — lightweight fallback

All models inherit from `models.base_model.BaseSLM` and implement:
```python
def forward(self, frames, context=None) -> ActionOutput
```

## Roadmap

- [x] Project structure & configs
- [ ] Screen capture + input logger
- [ ] Dataset builder (HDF5/Zarr)
- [ ] Base model + walking model
- [ ] Behavioral cloning trainer
- [ ] Car, bike, plane, helicopter models
- [ ] Real-time inference loop
- [ ] Mode auto-detection
- [ ] ONNX export & quantization
- [ ] DAgger fine-tuning

## License

Apache 2.0 — see [LICENSE](LICENSE).

"""Configuration system using dataclasses with YAML support."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    resolution: int = 328
    n_channels: int = 6
    channels: list = field(default_factory=lambda: [32, 64, 128, 256, 512])
    asr_mode: str = "ave"

    @property
    def n_down_layers(self) -> int:
        if self.resolution >= 256:
            return 5
        return 4

    @property
    def crop_size(self) -> int:
        if self.resolution >= 256:
            return 328
        return 168

    @property
    def inner_size(self) -> int:
        if self.resolution >= 256:
            return 320
        return 160

    @property
    def inner_crop(self) -> tuple:
        if self.resolution >= 256:
            return (4, 324)
        return (4, 164)

    @property
    def mask_rect(self) -> tuple:
        if self.resolution >= 256:
            return (5, 5, 310, 305)
        return (5, 5, 150, 145)

    @property
    def audio_feat_shape(self) -> dict:
        shapes = {
            "ave": (32, 16, 16),
            "hubert": (32, 32, 32),
            "wenet": (256, 16, 32),
        }
        return shapes[self.asr_mode]


@dataclass
class TrainConfig:
    """Training configuration."""
    epochs: int = 100
    batch_size: int = 8
    lr: float = 1e-3
    num_workers: int = 8
    use_syncnet: bool = True
    syncnet_checkpoint: str = ""
    dataset_dir: str = ""
    save_dir: str = "./checkpoint"
    save_interval: int = 5
    see_res: bool = False

    loss_pixel_weight: float = 1.0
    loss_perceptual_weight: float = 0.01
    loss_sync_weight: float = 10.0

    use_amp: bool = True
    lr_scheduler: str = "cosine"
    lr_warmup_epochs: int = 5
    lr_min: float = 1e-6

    resume_checkpoint: Optional[str] = None
    log_dir: str = "./logs"

    syncnet_epochs: int = 100
    syncnet_batch_size: int = 16
    syncnet_lr: float = 1e-3
    syncnet_num_workers: int = 16


@dataclass
class InferenceConfig:
    """Inference configuration."""
    batch_size: int = 4
    device: str = "auto"
    output_codec: str = "libx264"
    output_crf: int = 20
    fps: int = 25


@dataclass
class DataConfig:
    """Data preprocessing configuration."""
    sample_rate: int = 16000
    fps: int = 25
    n_fft: int = 800
    hop_size: int = 200
    win_size: int = 800
    n_mels: int = 80
    fmin: int = 55
    fmax: int = 7600
    preemphasis: float = 0.97
    ref_level_db: int = 20
    min_level_db: int = -100
    max_abs_value: float = 4.0


@dataclass
class SyncTalkConfig:
    """Root configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> "SyncTalkConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(
            model=ModelConfig(**data.get("model", {})),
            train=TrainConfig(**data.get("train", {})),
            inference=InferenceConfig(**data.get("inference", {})),
            data=DataConfig(**data.get("data", {})),
        )

    @classmethod
    def from_resolution(cls, resolution: int = 328, **kwargs) -> "SyncTalkConfig":
        config = cls()
        config.model.resolution = resolution
        for key, value in kwargs.items():
            parts = key.split(".")
            if len(parts) == 2:
                sub_config = getattr(config, parts[0])
                setattr(sub_config, parts[1], value)
            else:
                for sub in [config.model, config.train, config.inference, config.data]:
                    if hasattr(sub, key):
                        setattr(sub, key, value)
                        break
        return config

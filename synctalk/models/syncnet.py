"""Unified SyncNet audio-visual synchronization discriminator.

Merges syncnet.py (160px) and syncnet_328.py (328px) into a single
parameterized model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import Conv2dBN


class SyncNet(nn.Module):
    """Audio-visual synchronization network.

    Args:
        mode: Audio feature backend ("ave", "hubert", "wenet").
        resolution: Target resolution (160 or 328). Controls face_encoder depth.
    """

    def __init__(self, mode: str = "ave", resolution: int = 328):
        super().__init__()

        face_layers = [
            Conv2dBN(3, 32, kernel_size=(7, 7), stride=1, padding=3),
        ]

        if resolution >= 256:
            face_layers.append(
                Conv2dBN(32, 32, kernel_size=5, stride=2, padding=1)
            )

        face_layers.extend([
            Conv2dBN(32, 64, kernel_size=5, stride=2, padding=1),
            Conv2dBN(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dBN(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2dBN(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2dBN(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dBN(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dBN(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2dBN(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2dBN(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dBN(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2dBN(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2dBN(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dBN(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2dBN(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2dBN(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2dBN(512, 512, kernel_size=1, stride=1, padding=0),
        ])

        self.face_encoder = nn.Sequential(*face_layers)

        p1, p2 = self._get_audio_params(mode)
        self.audio_encoder = nn.Sequential(
            Conv2dBN(p1, 128, kernel_size=3, stride=1, padding=1),
            Conv2dBN(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dBN(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2dBN(128, 256, kernel_size=3, stride=p2, padding=1),
            Conv2dBN(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dBN(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2dBN(256, 256, kernel_size=3, stride=2, padding=2),
            Conv2dBN(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dBN(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2dBN(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2dBN(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dBN(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2dBN(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2dBN(512, 512, kernel_size=1, stride=1, padding=0),
        )

    @staticmethod
    def _get_audio_params(mode: str):
        if mode == "wenet":
            return 128, (1, 2)
        elif mode == "hubert":
            return 32, (2, 2)
        elif mode == "ave":
            return 32, 1
        raise ValueError(f"Unsupported mode: {mode}")

    def forward(self, face_sequences, audio_sequences):
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        return audio_embedding, face_embedding

    @classmethod
    def from_config(cls, config) -> "SyncNet":
        return cls(mode=config.asr_mode, resolution=config.resolution)

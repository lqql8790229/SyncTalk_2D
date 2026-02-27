"""Unified AudioEncoder for audio-visual embedding extraction.

Used both during data preprocessing and inference for extracting
audio features from mel spectrograms.
"""

import torch
import torch.nn as nn
from .blocks import Conv2dBN


class AudioEncoder(nn.Module):
    """Audio encoder that converts mel spectrograms to audio-visual embeddings.

    Input: [B, 1, 16, 80] mel spectrogram
    Output: [B, 512] embedding vector
    """

    def __init__(self):
        super().__init__()
        self.audio_encoder = nn.Sequential(
            Conv2dBN(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2dBN(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dBN(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2dBN(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2dBN(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dBN(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2dBN(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2dBN(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dBN(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2dBN(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2dBN(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2dBN(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2dBN(512, 512, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        out = self.audio_encoder(x)
        return out.squeeze(2).squeeze(2)

    def load_pretrained(self, checkpoint_path: str, device: torch.device = None):
        """Load pretrained weights with proper key mapping."""
        map_location = device or torch.device("cpu")
        ckpt = torch.load(checkpoint_path, map_location=map_location)
        new_state_dict = {f"audio_encoder.{k}": v for k, v in ckpt.items()}
        self.load_state_dict(new_state_dict)
        return self

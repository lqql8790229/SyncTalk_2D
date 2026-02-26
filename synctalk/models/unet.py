"""Unified UNet model supporting multiple resolutions.

Merges unet.py (160px, 4 down/up) and unet_328.py (328px, 5 down/up)
into a single parameterized architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import InvertedResidual, DoubleConvDW, InConvDw, Down, Up, OutConv


class AudioConvAve(nn.Module):
    """Audio feature convolution for AVE backend (input: [B, 32, 16, 16])."""

    def __init__(self, channels):
        super().__init__()
        ch = channels
        self.net = nn.Sequential(
            InvertedResidual(ch[0], ch[1], stride=1, use_res_connect=False, expand_ratio=2),
            InvertedResidual(ch[1], ch[2], stride=1, use_res_connect=False, expand_ratio=2),
            nn.Conv2d(ch[2], ch[3], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(ch[3]),
            nn.ReLU(inplace=True),
            InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True, expand_ratio=2),
            nn.Conv2d(ch[3], ch[4], kernel_size=3, padding=3, stride=2),
            nn.BatchNorm2d(ch[4]),
            nn.ReLU(inplace=True),
            InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2),
            InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2),
        )

    def forward(self, x):
        return self.net(x)


class AudioConvHubert(nn.Module):
    """Audio feature convolution for HuBERT backend (input: [B, 32, 32, 32])."""

    def __init__(self, channels):
        super().__init__()
        ch = channels
        self.conv1 = InvertedResidual(ch[0], ch[1], stride=1, use_res_connect=False, expand_ratio=2)
        self.conv2 = InvertedResidual(ch[1], ch[2], stride=1, use_res_connect=False, expand_ratio=2)
        self.conv3 = nn.Conv2d(ch[2], ch[3], kernel_size=3, padding=1, stride=(2, 2))
        self.bn3 = nn.BatchNorm2d(ch[3])
        self.conv4 = InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True, expand_ratio=2)
        self.conv5 = nn.Conv2d(ch[3], ch[4], kernel_size=3, padding=3, stride=2)
        self.bn5 = nn.BatchNorm2d(ch[4])
        self.relu = nn.ReLU(inplace=True)
        self.conv6 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2)
        self.conv7 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        return self.conv7(x)


class AudioConvWenet(nn.Module):
    """Audio feature convolution for WeNet backend (input: [B, 256, 16, 32])."""

    def __init__(self, channels):
        super().__init__()
        ch = channels
        self.conv1 = InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True, expand_ratio=2)
        self.conv2 = InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True, expand_ratio=2)
        self.conv3 = nn.Conv2d(ch[3], ch[3], kernel_size=3, padding=1, stride=(1, 2))
        self.bn3 = nn.BatchNorm2d(ch[3])
        self.conv4 = InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True, expand_ratio=2)
        self.conv5 = nn.Conv2d(ch[3], ch[4], kernel_size=3, padding=3, stride=2)
        self.bn5 = nn.BatchNorm2d(ch[4])
        self.relu = nn.ReLU(inplace=True)
        self.conv6 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2)
        self.conv7 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        return self.conv7(x)


_AUDIO_CONV_MAP = {
    "ave": AudioConvAve,
    "hubert": AudioConvHubert,
    "wenet": AudioConvWenet,
}


class UNet(nn.Module):
    """Unified UNet for lip-sync generation.

    Supports both 160px (4 down/up layers) and 328px (5 down/up layers)
    via the `n_down_layers` parameter.

    Args:
        n_channels: Number of input channels (default 6: reference + masked).
        mode: Audio feature backend ("ave", "hubert", "wenet").
        channels: Channel multipliers [ch0, ch1, ch2, ch3, ch4].
        n_down_layers: Number of downsampling layers (4 for 160px, 5 for 328px).
    """

    def __init__(self, n_channels: int = 6, mode: str = "ave",
                 channels: list = None, n_down_layers: int = 5):
        super().__init__()
        ch = channels or [32, 64, 128, 256, 512]
        self.n_down_layers = n_down_layers

        audio_conv_cls = _AUDIO_CONV_MAP.get(mode)
        if audio_conv_cls is None:
            raise ValueError(f"Unsupported audio mode: {mode}. Choose from {list(_AUDIO_CONV_MAP.keys())}")
        self.audio_model = audio_conv_cls(ch)

        self.fuse_conv = nn.Sequential(
            DoubleConvDW(ch[4] * 2, ch[4], stride=1),
            DoubleConvDW(ch[4], ch[3], stride=1),
        )

        if n_down_layers == 5:
            self.inc = InConvDw(n_channels, ch[0])
            self.down1 = Down(ch[0], ch[0])
            self.down2 = Down(ch[0], ch[1])
            self.down3 = Down(ch[1], ch[2])
            self.down4 = Down(ch[2], ch[3])
            self.down5 = Down(ch[3], ch[4])

            self.up1 = Up(ch[4], ch[3] // 2)
            self.up2 = Up(ch[3], ch[2] // 2)
            self.up3 = Up(ch[2], ch[1] // 2)
            self.up4 = Up(ch[1], ch[0])
            self.up5 = Up(ch[1], ch[0] // 2)
            self.outc = OutConv(ch[0] // 2, 3)
        elif n_down_layers == 4:
            self.inc = InConvDw(n_channels, ch[0])
            self.down1 = Down(ch[0], ch[1])
            self.down2 = Down(ch[1], ch[2])
            self.down3 = Down(ch[2], ch[3])
            self.down4 = Down(ch[3], ch[4])

            self.up1 = Up(ch[4], ch[3] // 2)
            self.up2 = Up(ch[3], ch[2] // 2)
            self.up3 = Up(ch[2], ch[1] // 2)
            self.up4 = Up(ch[1], ch[0])
            self.outc = OutConv(ch[0], 3)
        else:
            raise ValueError(f"n_down_layers must be 4 or 5, got {n_down_layers}")

    def forward(self, x, audio_feat):
        if self.n_down_layers == 5:
            return self._forward_5(x, audio_feat)
        return self._forward_4(x, audio_feat)

    def _forward_5(self, x, audio_feat):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        audio_feat = self.audio_model(audio_feat)
        x6 = torch.cat([x6, audio_feat], dim=1)
        x6 = self.fuse_conv(x6)

        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        return torch.sigmoid(self.outc(x))

    def _forward_4(self, x, audio_feat):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        audio_feat = self.audio_model(audio_feat)
        x5 = torch.cat([x5, audio_feat], dim=1)
        x5 = self.fuse_conv(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return torch.sigmoid(self.outc(x))

    @classmethod
    def from_config(cls, config) -> "UNet":
        return cls(
            n_channels=config.n_channels,
            mode=config.asr_mode,
            channels=config.channels,
            n_down_layers=config.n_down_layers,
        )

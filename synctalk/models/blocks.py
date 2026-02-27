"""Shared building blocks for all models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertedResidual(nn.Module):
    """MobileNetV2-style inverted residual block."""

    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super().__init__()
        assert stride in [1, 2]
        self.use_res_connect = use_res_connect
        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1,
                      groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class DoubleConvDW(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.double_conv = nn.Sequential(
            InvertedResidual(in_channels, out_channels, stride=stride,
                             use_res_connect=False, expand_ratio=2),
            InvertedResidual(out_channels, out_channels, stride=1,
                             use_res_connect=True, expand_ratio=2),
        )

    def forward(self, x):
        return self.double_conv(x)


class InConvDw(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inconv = InvertedResidual(
            in_channels, out_channels, stride=1,
            use_res_connect=False, expand_ratio=2,
        )

    def forward(self, x):
        return self.inconv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = DoubleConvDW(in_channels, out_channels, stride=2)

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConvDW(in_channels, out_channels, stride=1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.shape[2] - x1.shape[2]
        diff_x = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                         diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Conv2dBN(nn.Module):
    """Conv2d + BatchNorm + activation block for SyncNet."""

    def __init__(self, cin, cout, kernel_size, stride, padding,
                 residual=False, leaky_relu=False):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout),
        )
        if leaky_relu:
            self.act = nn.LeakyReLU(0.02)
        else:
            self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

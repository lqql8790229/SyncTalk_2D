"""Evaluation metrics for lip-sync quality assessment."""

import torch
import torch.nn.functional as F
import numpy as np


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio.

    Args:
        pred: Predicted image [B, C, H, W] in [0, 1].
        target: Ground truth image [B, C, H, W] in [0, 1].
        max_val: Maximum pixel value.

    Returns:
        Average PSNR in dB.
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float("inf")
    return (10 * torch.log10(max_val ** 2 / mse)).item()


def ssim(pred: torch.Tensor, target: torch.Tensor,
         window_size: int = 11, c1: float = 0.01 ** 2, c2: float = 0.03 ** 2) -> float:
    """Structural Similarity Index Measure.

    Simplified single-scale SSIM computation.

    Args:
        pred: Predicted image [B, C, H, W] in [0, 1].
        target: Ground truth image [B, C, H, W] in [0, 1].
        window_size: Size of the Gaussian window.
        c1, c2: Stability constants.

    Returns:
        Average SSIM score.
    """
    channels = pred.shape[1]
    kernel = _gaussian_kernel(window_size, 1.5).to(pred.device)
    kernel = kernel.expand(channels, 1, window_size, window_size)

    mu1 = F.conv2d(pred, kernel, groups=channels, padding=window_size // 2)
    mu2 = F.conv2d(target, kernel, groups=channels, padding=window_size // 2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred ** 2, kernel, groups=channels, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, kernel, groups=channels, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, groups=channels, padding=window_size // 2) - mu1_mu2

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = numerator / denominator
    return ssim_map.mean().item()


def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    """Create a 2D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = g.unsqueeze(0) * g.unsqueeze(1)
    return (kernel / kernel.sum()).unsqueeze(0).unsqueeze(0)


def landmark_distance(pred_landmarks: np.ndarray, gt_landmarks: np.ndarray) -> float:
    """Landmark Distance (LMD) between predicted and ground truth landmarks.

    Args:
        pred_landmarks: [N, 2] predicted landmark coordinates.
        gt_landmarks: [N, 2] ground truth landmark coordinates.

    Returns:
        Mean Euclidean distance across all landmarks.
    """
    return np.mean(np.sqrt(np.sum((pred_landmarks - gt_landmarks) ** 2, axis=1)))


class MetricsTracker:
    """Track and aggregate training/validation metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._metrics = {}
        self._counts = {}

    def update(self, name: str, value: float, count: int = 1):
        if name not in self._metrics:
            self._metrics[name] = 0.0
            self._counts[name] = 0
        self._metrics[name] += value * count
        self._counts[name] += count

    def get(self, name: str) -> float:
        if name not in self._metrics or self._counts[name] == 0:
            return 0.0
        return self._metrics[name] / self._counts[name]

    def summary(self) -> dict:
        return {name: self.get(name) for name in self._metrics}

    def __str__(self) -> str:
        parts = [f"{name}={self.get(name):.4f}" for name in sorted(self._metrics)]
        return " | ".join(parts)

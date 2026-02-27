"""Data augmentation pipeline for training."""

import random
import cv2
import numpy as np
import torch


class TrainAugmentation:
    """Composable data augmentation for lip-sync training.

    Applies augmentations that preserve lip-sync quality while improving
    model generalization.

    Args:
        color_jitter: Enable color jitter (brightness, contrast, saturation).
        horizontal_flip: Enable random horizontal flipping.
        gaussian_noise: Enable random Gaussian noise.
        brightness_range: (min, max) brightness multiplier.
        contrast_range: (min, max) contrast multiplier.
        saturation_range: (min, max) saturation multiplier.
        noise_std: Standard deviation for Gaussian noise.
        p_color: Probability of applying color jitter.
        p_flip: Probability of horizontal flip.
        p_noise: Probability of adding noise.
    """

    def __init__(self, color_jitter: bool = True, horizontal_flip: bool = True,
                 gaussian_noise: bool = True,
                 brightness_range: tuple = (0.8, 1.2),
                 contrast_range: tuple = (0.8, 1.2),
                 saturation_range: tuple = (0.8, 1.2),
                 noise_std: float = 0.02,
                 p_color: float = 0.5, p_flip: float = 0.5, p_noise: float = 0.3):
        self.color_jitter = color_jitter
        self.horizontal_flip = horizontal_flip
        self.gaussian_noise = gaussian_noise
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.noise_std = noise_std
        self.p_color = p_color
        self.p_flip = p_flip
        self.p_noise = p_noise

    def _apply_color_jitter(self, img: np.ndarray) -> np.ndarray:
        """Apply random color jitter to BGR image."""
        img = img.astype(np.float32)
        brightness = random.uniform(*self.brightness_range)
        img = np.clip(img * brightness, 0, 255)

        contrast = random.uniform(*self.contrast_range)
        mean = img.mean()
        img = np.clip((img - mean) * contrast + mean, 0, 255)

        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        saturation = random.uniform(*self.saturation_range)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return img.astype(np.uint8)

    def augment_image(self, img: np.ndarray) -> np.ndarray:
        """Apply augmentations to a single BGR image (numpy)."""
        if self.color_jitter and random.random() < self.p_color:
            img = self._apply_color_jitter(img)

        if self.horizontal_flip and random.random() < self.p_flip:
            img = cv2.flip(img, 1)

        return img

    def augment_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to a tensor [C, H, W] in range [0, 1]."""
        if self.gaussian_noise and random.random() < self.p_noise:
            noise = torch.randn_like(tensor) * self.noise_std
            tensor = torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor

    def augment_pair(self, img: np.ndarray, label: np.ndarray):
        """Apply consistent augmentation to an image and its label.

        Returns:
            (augmented_img, augmented_label)
        """
        do_flip = self.horizontal_flip and random.random() < self.p_flip
        do_color = self.color_jitter and random.random() < self.p_color

        if do_color:
            img = self._apply_color_jitter(img)
            label = self._apply_color_jitter(label)

        if do_flip:
            img = cv2.flip(img, 1)
            label = cv2.flip(label, 1)

        return img, label

"""Tests for data augmentation."""

import numpy as np
import torch
import pytest
from synctalk.data.augmentation import TrainAugmentation


class TestTrainAugmentation:
    def test_create(self):
        aug = TrainAugmentation()
        assert aug.color_jitter
        assert aug.horizontal_flip
        assert aug.gaussian_noise

    def test_augment_image(self):
        aug = TrainAugmentation(p_color=1.0, p_flip=0.0)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = aug.augment_image(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_augment_tensor(self):
        aug = TrainAugmentation(p_noise=1.0, noise_std=0.1)
        tensor = torch.rand(3, 64, 64)
        result = aug.augment_tensor(tensor)
        assert result.shape == tensor.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_augment_pair(self):
        aug = TrainAugmentation(p_color=1.0, p_flip=1.0)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        label = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        aug_img, aug_label = aug.augment_pair(img, label)
        assert aug_img.shape == img.shape
        assert aug_label.shape == label.shape

    def test_no_augmentation(self):
        aug = TrainAugmentation(color_jitter=False, horizontal_flip=False,
                                gaussian_noise=False)
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        result = aug.augment_image(img)
        np.testing.assert_array_equal(result, img)

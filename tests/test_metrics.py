"""Tests for evaluation metrics."""

import torch
import numpy as np
import pytest
from synctalk.training.metrics import psnr, ssim, landmark_distance, MetricsTracker


class TestPSNR:
    def test_identical_images(self):
        img = torch.rand(1, 3, 64, 64)
        assert psnr(img, img) == float("inf")

    def test_different_images(self):
        pred = torch.rand(1, 3, 64, 64)
        target = torch.rand(1, 3, 64, 64)
        val = psnr(pred, target)
        assert 0 < val < 50

    def test_noisy_image(self):
        target = torch.rand(1, 3, 64, 64)
        pred = target + torch.randn_like(target) * 0.01
        pred = torch.clamp(pred, 0, 1)
        val = psnr(pred, target)
        assert val > 30


class TestSSIM:
    def test_identical_images(self):
        img = torch.rand(1, 3, 32, 32)
        val = ssim(img, img)
        assert abs(val - 1.0) < 0.01

    def test_different_images(self):
        pred = torch.rand(1, 3, 32, 32)
        target = torch.rand(1, 3, 32, 32)
        val = ssim(pred, target)
        assert -1 <= val <= 1


class TestLandmarkDistance:
    def test_identical(self):
        lms = np.array([[100, 200], [150, 250]], dtype=np.float32)
        assert landmark_distance(lms, lms) == 0.0

    def test_known_distance(self):
        pred = np.array([[0, 0], [3, 4]], dtype=np.float32)
        gt = np.array([[0, 0], [0, 0]], dtype=np.float32)
        assert abs(landmark_distance(pred, gt) - 2.5) < 0.01


class TestMetricsTracker:
    def test_basic(self):
        tracker = MetricsTracker()
        tracker.update("loss", 0.5)
        tracker.update("loss", 0.3)
        assert abs(tracker.get("loss") - 0.4) < 1e-6

    def test_summary(self):
        tracker = MetricsTracker()
        tracker.update("psnr", 30.0)
        tracker.update("ssim", 0.9)
        s = tracker.summary()
        assert "psnr" in s
        assert "ssim" in s

    def test_reset(self):
        tracker = MetricsTracker()
        tracker.update("loss", 1.0)
        tracker.reset()
        assert tracker.get("loss") == 0.0

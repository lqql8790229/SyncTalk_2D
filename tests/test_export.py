"""Tests for model export utilities."""

import pytest
import torch
from synctalk.models.export import get_model_info
from synctalk.models import UNet, AudioEncoder


class TestGetModelInfo:
    def test_unet_info(self):
        model = UNet(6, "ave", n_down_layers=5)
        info = get_model_info(model)
        assert info["total_params"] == 12_186_935
        assert info["trainable_params"] == 12_186_935
        assert info["total_size_mb"] > 0

    def test_audio_encoder_info(self):
        model = AudioEncoder()
        info = get_model_info(model)
        assert info["total_params"] == 2_812_672

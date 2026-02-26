"""Tests for configuration system."""

import os
import tempfile
import pytest
from synctalk.configs import ModelConfig, TrainConfig, InferenceConfig, SyncTalkConfig


class TestModelConfig:
    def test_default_328(self):
        cfg = ModelConfig(resolution=328)
        assert cfg.n_down_layers == 5
        assert cfg.crop_size == 328
        assert cfg.inner_crop == (4, 324)
        assert cfg.mask_rect == (5, 5, 310, 305)
        assert cfg.inner_size == 320

    def test_default_160(self):
        cfg = ModelConfig(resolution=160)
        assert cfg.n_down_layers == 4
        assert cfg.crop_size == 168
        assert cfg.inner_crop == (4, 164)
        assert cfg.mask_rect == (5, 5, 150, 145)
        assert cfg.inner_size == 160

    def test_audio_feat_shapes(self):
        for mode, expected in [("ave", (32, 16, 16)),
                               ("hubert", (32, 32, 32)),
                               ("wenet", (256, 16, 32))]:
            cfg = ModelConfig(asr_mode=mode)
            assert cfg.audio_feat_shape == expected


class TestSyncTalkConfig:
    def test_from_resolution(self):
        cfg = SyncTalkConfig.from_resolution(328)
        assert cfg.model.resolution == 328

        cfg = SyncTalkConfig.from_resolution(160)
        assert cfg.model.resolution == 160

    def test_from_resolution_kwargs(self):
        cfg = SyncTalkConfig.from_resolution(328, asr_mode="hubert")
        assert cfg.model.asr_mode == "hubert"

    def test_yaml_roundtrip(self):
        cfg = SyncTalkConfig.from_resolution(328, asr_mode="ave")
        cfg.train.epochs = 50
        cfg.train.lr = 0.0005

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name

        try:
            cfg.save(path)
            loaded = SyncTalkConfig.load(path)
            assert loaded.model.resolution == 328
            assert loaded.model.asr_mode == "ave"
            assert loaded.train.epochs == 50
            assert loaded.train.lr == 0.0005
        finally:
            os.unlink(path)

    def test_preset_files_exist(self):
        preset_dir = os.path.join(os.path.dirname(__file__),
                                   "..", "synctalk", "configs", "presets")
        assert os.path.exists(os.path.join(preset_dir, "160.yaml"))
        assert os.path.exists(os.path.join(preset_dir, "328.yaml"))

    def test_load_preset(self):
        preset_path = os.path.join(os.path.dirname(__file__),
                                    "..", "synctalk", "configs", "presets", "328.yaml")
        cfg = SyncTalkConfig.load(preset_path)
        assert cfg.model.resolution == 328

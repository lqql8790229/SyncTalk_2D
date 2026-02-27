"""Tests for model architectures."""

import pytest
import torch
from synctalk.models import UNet, SyncNet, AudioEncoder
from synctalk.configs import ModelConfig


class TestUNet:
    def test_create_328(self):
        model = UNet(6, "ave", n_down_layers=5)
        assert sum(p.numel() for p in model.parameters()) == 12_186_935

    def test_create_160(self):
        model = UNet(6, "ave", n_down_layers=4)
        assert sum(p.numel() for p in model.parameters()) == 12_163_591

    def test_forward_328(self):
        model = UNet(6, "ave", n_down_layers=5)
        x = torch.randn(1, 6, 320, 320)
        a = torch.randn(1, 32, 16, 16)
        with torch.no_grad():
            out = model(x, a)
        assert out.shape == (1, 3, 320, 320)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_forward_160(self):
        model = UNet(6, "ave", n_down_layers=4)
        x = torch.randn(1, 6, 160, 160)
        a = torch.randn(1, 32, 16, 16)
        with torch.no_grad():
            out = model(x, a)
        assert out.shape == (1, 3, 160, 160)

    def test_batch_forward(self):
        model = UNet(6, "ave", n_down_layers=5)
        x = torch.randn(4, 6, 320, 320)
        a = torch.randn(4, 32, 16, 16)
        with torch.no_grad():
            out = model(x, a)
        assert out.shape == (4, 3, 320, 320)

    def test_from_config(self):
        cfg = ModelConfig(resolution=328, asr_mode="ave")
        model = UNet.from_config(cfg)
        assert sum(p.numel() for p in model.parameters()) == 12_186_935

    def test_all_modes(self):
        for mode in ["ave", "hubert", "wenet"]:
            model = UNet(6, mode, n_down_layers=5)
            assert model is not None

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Unsupported"):
            UNet(6, "invalid_mode")

    def test_invalid_n_down(self):
        with pytest.raises(ValueError, match="n_down_layers"):
            UNet(6, "ave", n_down_layers=3)

    @pytest.mark.parametrize("mode,audio_shape", [
        ("ave", (1, 32, 16, 16)),
        ("hubert", (1, 32, 32, 32)),
        ("wenet", (1, 256, 16, 32)),
    ])
    def test_audio_modes_forward(self, mode, audio_shape):
        model = UNet(6, mode, n_down_layers=5)
        x = torch.randn(1, 6, 320, 320)
        a = torch.randn(*audio_shape)
        with torch.no_grad():
            out = model(x, a)
        assert out.shape == (1, 3, 320, 320)

    def test_equivalence_with_old_328(self):
        """Verify parameter count matches original unet_328.py."""
        import sys
        sys.path.insert(0, "/workspace")
        from unet_328 import Model as OldModel
        old = OldModel(6, "ave")
        new = UNet(6, "ave", n_down_layers=5)
        assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters())

    def test_equivalence_with_old_160(self):
        """Verify parameter count matches original unet.py."""
        import sys
        sys.path.insert(0, "/workspace")
        from unet import Model as OldModel
        old = OldModel(6, "ave")
        new = UNet(6, "ave", n_down_layers=4)
        assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters())


class TestSyncNet:
    def test_create_328(self):
        model = SyncNet("ave", resolution=328)
        assert model is not None

    def test_create_160(self):
        model = SyncNet("ave", resolution=160)
        assert model is not None

    def test_328_has_more_params(self):
        m328 = SyncNet("ave", resolution=328)
        m160 = SyncNet("ave", resolution=160)
        p328 = sum(p.numel() for p in m328.parameters())
        p160 = sum(p.numel() for p in m160.parameters())
        assert p328 > p160

    def test_forward(self):
        model = SyncNet("ave", resolution=328)
        face = torch.randn(2, 3, 320, 320)
        audio = torch.randn(2, 32, 16, 16)
        with torch.no_grad():
            a_emb, f_emb = model(face, audio)
        assert a_emb.shape[0] == 2
        assert f_emb.shape[0] == 2
        assert torch.allclose(a_emb.norm(dim=1), torch.ones(2), atol=1e-5)
        assert torch.allclose(f_emb.norm(dim=1), torch.ones(2), atol=1e-5)

    def test_from_config(self):
        cfg = ModelConfig(resolution=328, asr_mode="ave")
        model = SyncNet.from_config(cfg)
        assert model is not None


class TestAudioEncoder:
    def test_create(self):
        model = AudioEncoder()
        assert sum(p.numel() for p in model.parameters()) == 2_812_672

    def test_forward(self):
        model = AudioEncoder().eval()
        x = torch.randn(4, 1, 80, 16)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 512)

    def test_load_pretrained(self):
        model = AudioEncoder()
        model.load_pretrained("model/checkpoints/audio_visual_encoder.pth")
        model.eval()
        x = torch.randn(1, 1, 80, 16)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 512)
        assert not torch.all(out == 0)

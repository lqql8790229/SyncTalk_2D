"""Tests for audio processing."""

import os
import pytest
import torch
import numpy as np
from synctalk.data.audio import (
    load_wav, melspectrogram, get_audio_features, AudDataset,
)


DEMO_WAV = "demo/talk_hb.wav"


@pytest.fixture
def demo_wav_exists():
    if not os.path.exists(DEMO_WAV):
        pytest.skip("Demo WAV not found")
    return DEMO_WAV


class TestMelSpectrogram:
    def test_load_wav(self, demo_wav_exists):
        wav = load_wav(demo_wav_exists, 16000)
        assert isinstance(wav, np.ndarray)
        assert len(wav) > 0

    def test_melspectrogram(self, demo_wav_exists):
        wav = load_wav(demo_wav_exists, 16000)
        mel = melspectrogram(wav)
        assert isinstance(mel, np.ndarray)
        assert mel.shape[0] == 80
        assert mel.shape[1] > 0


class TestGetAudioFeatures:
    def test_middle_index(self):
        features = np.random.randn(100, 512).astype(np.float32)
        feat = get_audio_features(features, 50)
        assert feat.shape == (16, 512)

    def test_start_padding(self):
        features = np.random.randn(100, 512).astype(np.float32)
        feat = get_audio_features(features, 0)
        assert feat.shape == (16, 512)

    def test_end_padding(self):
        features = np.random.randn(100, 512).astype(np.float32)
        feat = get_audio_features(features, 99)
        assert feat.shape == (16, 512)


class TestAudDataset:
    def test_create(self, demo_wav_exists):
        ds = AudDataset(demo_wav_exists)
        assert len(ds) > 0

    def test_getitem(self, demo_wav_exists):
        ds = AudDataset(demo_wav_exists)
        mel = ds[0]
        assert mel.shape == (1, 80, 16)

    def test_full_pipeline(self, demo_wav_exists):
        from synctalk.models import AudioEncoder
        from torch.utils.data import DataLoader

        ds = AudDataset(demo_wav_exists)
        dl = DataLoader(ds, batch_size=64, shuffle=False)
        ae = AudioEncoder().eval()
        ae.load_pretrained("model/checkpoints/audio_visual_encoder.pth")

        outputs = []
        for mel in dl:
            with torch.no_grad():
                outputs.append(ae(mel))
        result = torch.cat(outputs, dim=0)
        assert result.shape == (len(ds), 512)

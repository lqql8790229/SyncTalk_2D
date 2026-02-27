"""Tests for real-time pipeline modules."""

import pytest
import numpy as np
import torch
from synctalk.realtime.audio_stream import AudioStreamProcessor
from synctalk.realtime.virtual_camera import VirtualCameraOutput, PreviewWindow
from synctalk.realtime.character import Character
from synctalk.realtime.pipeline import RealtimePipeline


class TestAudioStreamProcessor:
    def test_create(self):
        proc = AudioStreamProcessor(device=torch.device("cpu"))
        assert proc.sample_rate == 16000
        assert proc.fps == 25

    def test_feed_audio(self):
        proc = AudioStreamProcessor(device=torch.device("cpu"))
        chunk = np.random.randn(640).astype(np.float32)
        proc.feed_audio(chunk)
        assert len(proc._audio_buffer) == 640

    def test_extract_feature_empty(self):
        proc = AudioStreamProcessor(device=torch.device("cpu"))
        feat = proc.extract_feature()
        assert feat.shape == (512,)
        assert np.all(feat == 0)

    def test_load_audio_file(self):
        proc = AudioStreamProcessor(device=torch.device("cpu"))
        features = proc.load_audio_file("demo/talk_hb.wav")
        assert features.shape[1] == 512
        assert features.shape[0] > 100

    def test_get_windowed_features(self):
        proc = AudioStreamProcessor(device=torch.device("cpu"))
        feats = proc.get_windowed_features(window=8)
        assert feats.shape == (16, 512)


class TestVirtualCameraOutput:
    def test_create(self):
        cam = VirtualCameraOutput(1280, 720, 25)
        assert cam.width == 1280
        assert cam.height == 720
        assert not cam.is_running

    def test_not_started_raises(self):
        cam = VirtualCameraOutput()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="not started"):
            cam.send_frame(frame)


class TestPreviewWindow:
    def test_create(self):
        preview = PreviewWindow("Test")
        assert preview.window_name == "Test"
        assert not preview._open


class TestCLILiveCommand:
    def test_live_command_registered(self):
        import subprocess
        result = subprocess.run(
            ["python", "-m", "synctalk.cli", "live", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--name" in result.stdout
        assert "--audio_file" in result.stdout
        assert "--mic_device" in result.stdout
        assert "--no_camera" in result.stdout

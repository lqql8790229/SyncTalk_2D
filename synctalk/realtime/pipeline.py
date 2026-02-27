"""Main real-time lip-sync pipeline.

Orchestrates the complete real-time workflow:
    Audio Input → Feature Extraction → UNet Inference → Virtual Camera

Supports two modes:
1. Live microphone mode: Real-time audio from mic
2. File playback mode: Pre-recorded audio file

Architecture:
    ┌─────────────┐     ┌──────────────┐     ┌───────────┐     ┌────────────────┐
    │ Microphone / │ ──→ │ AudioStream  │ ──→ │ Character │ ──→ │ Virtual Camera │
    │ Audio File   │     │ Processor    │     │ .render() │     │ / Preview      │
    └─────────────┘     └──────────────┘     └───────────┘     └────────────────┘
"""

import time
import logging
import threading
from typing import Optional

import numpy as np
import torch

from .audio_stream import AudioStreamProcessor
from .character import Character
from .virtual_camera import VirtualCameraOutput, PreviewWindow
from ..configs.base import ModelConfig, SyncTalkConfig
from ..data.audio import get_audio_features
from ..utils.device import get_device

logger = logging.getLogger(__name__)


class RealtimePipeline:
    """Complete real-time lip-sync pipeline.

    Args:
        character_name: Name of the pre-trained character.
        config: SyncTalkConfig (auto-created if None).
        device_str: Device ("auto", "cuda", "cuda:0", "cpu").
        camera_width: Virtual camera output width.
        camera_height: Virtual camera output height.
        fps: Target frame rate.
        enable_virtual_camera: Whether to output to virtual camera.
        enable_preview: Whether to show preview window.
    """

    def __init__(self, character_name: str,
                 config: SyncTalkConfig = None,
                 device_str: str = "auto",
                 camera_width: int = 1280,
                 camera_height: int = 720,
                 fps: int = 25,
                 enable_virtual_camera: bool = True,
                 enable_preview: bool = True):
        self.fps = fps
        self.enable_virtual_camera = enable_virtual_camera
        self.enable_preview = enable_preview
        self._running = False
        self._frame_idx = 0
        self._step_stride = 1

        cfg = config or SyncTalkConfig.from_resolution(328)
        self.device = get_device(device_str)

        logger.info(f"Loading character: {character_name} on {self.device}")
        self.character = Character(character_name, config=cfg.model, device=self.device)

        self.audio_processor = AudioStreamProcessor(device=self.device)

        self.virtual_camera = None
        if enable_virtual_camera:
            self.virtual_camera = VirtualCameraOutput(
                camera_width, camera_height, fps)

        self.preview = None
        if enable_preview:
            self.preview = PreviewWindow(f"SyncTalk - {character_name}")

        self._stats = {"fps": 0.0, "latency_ms": 0.0, "frames": 0}

    @property
    def stats(self) -> dict:
        return self._stats.copy()

    def run_with_audio_file(self, audio_path: str, loop: bool = False):
        """Run pipeline with a pre-recorded audio file.

        Args:
            audio_path: Path to WAV audio file.
            loop: Whether to loop the audio.
        """
        logger.info(f"Loading audio: {audio_path}")
        audio_features = self.audio_processor.load_audio_file(audio_path)
        total_frames = audio_features.shape[0]
        logger.info(f"Audio frames: {total_frames}, character frames: {self.character.n_frames}")

        if self.virtual_camera:
            self.virtual_camera.start()

        self._running = True
        self._frame_idx = 0
        frame_time = 1.0 / self.fps
        frame_count = 0
        fps_counter_start = time.perf_counter()

        try:
            while self._running:
                t0 = time.perf_counter()

                audio_idx = frame_count % total_frames if loop else frame_count
                if audio_idx >= total_frames:
                    if not loop:
                        logger.info("Audio playback complete")
                        break

                feat = get_audio_features(audio_features, audio_idx)
                output = self.character.render_frame(self._frame_idx, feat.numpy())

                if self.virtual_camera and self.virtual_camera.is_running:
                    self.virtual_camera.send_frame(output)

                if self.preview:
                    key = self.preview.show(output)
                    if key == ord('q') or key == 27:
                        logger.info("User pressed quit")
                        break

                self._advance_frame()
                frame_count += 1

                elapsed = time.perf_counter() - t0
                self._stats["latency_ms"] = elapsed * 1000

                sleep_time = frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if frame_count % 25 == 0:
                    now = time.perf_counter()
                    self._stats["fps"] = 25.0 / (now - fps_counter_start)
                    fps_counter_start = now
                    self._stats["frames"] = frame_count

        finally:
            self._cleanup()

    def run_with_microphone(self, mic_device_index: int = None):
        """Run pipeline with live microphone input.

        Args:
            mic_device_index: Microphone device index (None for default).
        """
        logger.info("Starting microphone capture...")
        self.audio_processor.start_microphone(mic_device_index)

        if self.virtual_camera:
            self.virtual_camera.start()

        self._running = True
        self._frame_idx = 0
        frame_time = 1.0 / self.fps
        frame_count = 0
        fps_counter_start = time.perf_counter()

        try:
            while self._running:
                t0 = time.perf_counter()

                feat = self.audio_processor.get_windowed_features(window=8)
                feat_tensor = torch.from_numpy(feat).float()

                output = self.character.render_frame(self._frame_idx, feat.flatten()[:512*16])

                if self.virtual_camera and self.virtual_camera.is_running:
                    self.virtual_camera.send_frame(output)

                if self.preview:
                    key = self.preview.show(output)
                    if key == ord('q') or key == 27:
                        break

                self._advance_frame()
                frame_count += 1

                elapsed = time.perf_counter() - t0
                self._stats["latency_ms"] = elapsed * 1000

                sleep_time = frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if frame_count % 25 == 0:
                    now = time.perf_counter()
                    self._stats["fps"] = 25.0 / (now - fps_counter_start)
                    fps_counter_start = now
                    self._stats["frames"] = frame_count

        finally:
            self.audio_processor.stop_microphone()
            self._cleanup()

    def _advance_frame(self):
        """Advance to next character frame (bounce at boundaries)."""
        if self._frame_idx >= self.character.n_frames - 1:
            self._step_stride = -1
        elif self._frame_idx <= 0:
            self._step_stride = 1
        self._frame_idx += self._step_stride

    def stop(self):
        """Signal the pipeline to stop."""
        self._running = False

    def _cleanup(self):
        """Clean up resources."""
        if self.virtual_camera:
            self.virtual_camera.stop()
        if self.preview:
            self.preview.close()
        logger.info(f"Pipeline stopped. Total frames: {self._stats['frames']}")

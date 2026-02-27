"""Real-time audio capture and streaming feature extraction.

Captures microphone audio in real-time, computes mel spectrograms
on a sliding window, and feeds them to AudioEncoder for feature
extraction. Designed for <10ms latency per frame.

Architecture:
    Microphone → Ring Buffer → Mel Spectrogram → AudioEncoder → Feature Queue
"""

import threading
import queue
import logging
import numpy as np
import torch
from collections import deque

from ..data.audio import melspectrogram, load_wav
from ..models.audio_encoder import AudioEncoder

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
HOP_SIZE = 200
MEL_WINDOW = 16
FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS  # 640 samples per frame at 25fps


class AudioStreamProcessor:
    """Real-time audio stream processor.

    Captures audio from microphone, extracts mel spectrograms in a sliding
    window, and produces audio feature embeddings in real-time.

    Args:
        encoder_checkpoint: Path to AudioEncoder weights.
        device: Torch device.
        sample_rate: Audio sample rate.
        fps: Target frame rate.
    """

    def __init__(self, encoder_checkpoint: str = "model/checkpoints/audio_visual_encoder.pth",
                 device: torch.device = None, sample_rate: int = 16000, fps: int = 25):
        self.sample_rate = sample_rate
        self.fps = fps
        self.samples_per_frame = sample_rate // fps
        self.device = device or torch.device("cpu")

        self.encoder = AudioEncoder().to(self.device).eval()
        self.encoder.load_pretrained(encoder_checkpoint, self.device)

        self._audio_buffer = deque(maxlen=sample_rate * 10)
        self._feature_buffer = deque(maxlen=200)
        self._feature_queue = queue.Queue(maxsize=50)
        self._running = False
        self._lock = threading.Lock()
        self._frame_count = 0

    def feed_audio(self, audio_chunk: np.ndarray):
        """Feed audio samples into the buffer.

        Args:
            audio_chunk: Float32 audio samples, mono, at self.sample_rate.
        """
        with self._lock:
            self._audio_buffer.extend(audio_chunk.flatten())

    def _compute_mel_for_frame(self, frame_idx: int) -> np.ndarray:
        """Compute mel spectrogram window for a specific frame."""
        with self._lock:
            audio_data = np.array(self._audio_buffer, dtype=np.float32)

        if len(audio_data) < self.samples_per_frame * 4:
            return None

        mel_full = melspectrogram(audio_data).T

        start_idx = int(80.0 * (frame_idx / float(self.fps)))
        end_idx = start_idx + MEL_WINDOW
        if end_idx > mel_full.shape[0]:
            end_idx = mel_full.shape[0]
            start_idx = max(0, end_idx - MEL_WINDOW)

        if end_idx - start_idx < MEL_WINDOW:
            return None

        return mel_full[start_idx:end_idx, :]

    @torch.no_grad()
    def extract_feature(self) -> np.ndarray:
        """Extract audio feature for the current frame.

        Returns:
            Feature embedding [512] or None if insufficient audio.
        """
        mel = self._compute_mel_for_frame(self._frame_count)
        if mel is None:
            if len(self._feature_buffer) > 0:
                return self._feature_buffer[-1]
            return np.zeros(512, dtype=np.float32)

        mel_tensor = torch.FloatTensor(mel.T).unsqueeze(0).unsqueeze(0).to(self.device)
        feat = self.encoder(mel_tensor).cpu().numpy().flatten()

        self._feature_buffer.append(feat)
        self._frame_count += 1
        return feat

    def get_windowed_features(self, window: int = 8) -> np.ndarray:
        """Get a windowed feature array for the current frame.

        Returns:
            Feature array [2*window, 512] for audio-visual fusion.
        """
        feat = self.extract_feature()
        feats = list(self._feature_buffer)

        if len(feats) < 2 * window:
            pad_count = 2 * window - len(feats)
            feats = [np.zeros_like(feat)] * pad_count + feats

        start = max(0, len(feats) - 2 * window)
        windowed = np.array(feats[start:start + 2 * window], dtype=np.float32)
        return windowed

    def start_microphone(self, device_index: int = None):
        """Start capturing from microphone in a background thread.

        Requires `sounddevice` package.
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("pip install sounddevice")

        self._running = True

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio callback status: {status}")
            self.feed_audio(indata[:, 0])

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.samples_per_frame,
            callback=callback,
            device=device_index,
        )
        self._stream.start()
        logger.info(f"Microphone capture started (sr={self.sample_rate}, fps={self.fps})")

    def stop_microphone(self):
        """Stop microphone capture."""
        self._running = False
        if hasattr(self, "_stream"):
            self._stream.stop()
            self._stream.close()
            logger.info("Microphone capture stopped")

    def load_audio_file(self, wav_path: str) -> np.ndarray:
        """Pre-load an audio file for offline/demo processing.

        Returns:
            Full audio feature array [N+2, 512].
        """
        from torch.utils.data import DataLoader
        from ..data.audio import AudDataset

        dataset = AudDataset(wav_path, self.sample_rate)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        outputs = []
        with torch.no_grad():
            for mel in loader:
                mel = mel.to(self.device)
                outputs.append(self.encoder(mel))

        outputs = torch.cat(outputs, dim=0).cpu()
        first, last = outputs[:1], outputs[-1:]
        full = torch.cat([first, outputs, last], dim=0).numpy()
        return full

"""File-based TTS engine (plays back a WAV file as if it were TTS output)."""

import logging
from typing import Callable

import numpy as np

from .base import TTSEngine
from ..data.audio import load_wav

logger = logging.getLogger(__name__)


class FileTTSEngine(TTSEngine):
    """TTS engine that plays back a WAV file.

    Useful for testing the pipeline without actual TTS.
    """

    def __init__(self, default_wav: str = None):
        self.default_wav = default_wav

    def synthesize(self, text: str, output_path: str) -> str:
        if self.default_wav:
            import shutil
            shutil.copy2(self.default_wav, output_path)
            return output_path
        raise NotImplementedError("FileTTSEngine requires a default_wav")

    def synthesize_to_stream(self, text: str,
                              callback: Callable[[np.ndarray], None],
                              sample_rate: int = 16000) -> None:
        if not self.default_wav:
            raise NotImplementedError("FileTTSEngine requires a default_wav")

        wav = load_wav(self.default_wav, sample_rate)
        chunk_size = sample_rate // 25
        for i in range(0, len(wav), chunk_size):
            callback(wav[i:i + chunk_size])

    def list_voices(self) -> list:
        return [{"name": "file", "gender": "N/A", "locale": "N/A"}]

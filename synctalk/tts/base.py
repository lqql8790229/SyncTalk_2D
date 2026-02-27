"""Abstract base class for TTS engines."""

from abc import ABC, abstractmethod
from typing import Callable, Optional
import numpy as np


class TTSEngine(ABC):
    """Abstract TTS engine interface.

    All TTS implementations must provide:
    1. synthesize() - text to WAV file
    2. synthesize_to_stream() - text to streaming PCM audio callback
    """

    @abstractmethod
    def synthesize(self, text: str, output_path: str) -> str:
        """Synthesize text to a WAV file.

        Args:
            text: Text to synthesize.
            output_path: Output WAV file path.

        Returns:
            Path to output WAV file.
        """
        ...

    @abstractmethod
    def synthesize_to_stream(self, text: str,
                              callback: Callable[[np.ndarray], None],
                              sample_rate: int = 16000) -> None:
        """Synthesize text and stream audio chunks via callback.

        Args:
            text: Text to synthesize.
            callback: Function called with each audio chunk (float32 numpy array).
            sample_rate: Target sample rate for output.
        """
        ...

    @abstractmethod
    def list_voices(self) -> list:
        """List available voices."""
        ...

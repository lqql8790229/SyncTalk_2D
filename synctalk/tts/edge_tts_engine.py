"""Microsoft Edge TTS engine.

Uses the edge-tts library for free, high-quality text-to-speech.
Supports streaming output for low-latency real-time pipeline.

Requires: pip install edge-tts
"""

import io
import asyncio
import logging
import tempfile
from typing import Callable, Optional

import numpy as np

from .base import TTSEngine

logger = logging.getLogger(__name__)

DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"

POPULAR_VOICES = {
    "xiaoxiao": "zh-CN-XiaoxiaoNeural",
    "xiaoyi": "zh-CN-XiaoyiNeural",
    "yunjian": "zh-CN-YunjianNeural",
    "yunxi": "zh-CN-YunxiNeural",
    "yunxia": "zh-CN-YunxiaNeural",
    "yunyang": "zh-CN-YunyangNeural",
    "jenny": "en-US-JennyNeural",
    "guy": "en-US-GuyNeural",
    "aria": "en-US-AriaNeural",
    "nanami": "ja-JP-NanamiNeural",
}


class EdgeTTSEngine(TTSEngine):
    """Microsoft Edge TTS engine (free, high quality, requires internet).

    Args:
        voice: Voice name or shorthand (e.g., "xiaoxiao", "zh-CN-XiaoxiaoNeural").
        rate: Speed adjustment (e.g., "+10%", "-20%", "+0%").
        volume: Volume adjustment (e.g., "+0%", "+50%").
    """

    def __init__(self, voice: str = DEFAULT_VOICE,
                 rate: str = "+0%", volume: str = "+0%"):
        self.voice = POPULAR_VOICES.get(voice, voice)
        self.rate = rate
        self.volume = volume
        self._loop = None

    def _get_loop(self):
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def synthesize(self, text: str, output_path: str) -> str:
        """Synthesize text to MP3 file, then convert to WAV."""
        import edge_tts

        async def _run():
            communicate = edge_tts.Communicate(
                text, self.voice, rate=self.rate, volume=self.volume
            )
            await communicate.save(output_path)

        loop = self._get_loop()
        loop.run_until_complete(_run())
        logger.info(f"TTS saved: {output_path}")
        return output_path

    def synthesize_to_stream(self, text: str,
                              callback: Callable[[np.ndarray], None],
                              sample_rate: int = 16000) -> None:
        """Synthesize text and stream audio chunks via callback."""
        import edge_tts

        audio_chunks = []

        async def _run():
            communicate = edge_tts.Communicate(
                text, self.voice, rate=self.rate, volume=self.volume
            )
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])

        loop = self._get_loop()
        loop.run_until_complete(_run())

        if not audio_chunks:
            logger.warning("TTS produced no audio")
            return

        raw_audio = b"".join(audio_chunks)
        pcm = self._decode_mp3_to_pcm(raw_audio, sample_rate)
        if pcm is not None:
            chunk_size = sample_rate // 25
            for i in range(0, len(pcm), chunk_size):
                callback(pcm[i:i + chunk_size])

    def synthesize_to_wav(self, text: str, output_path: str,
                           sample_rate: int = 16000) -> str:
        """Synthesize text directly to a 16kHz WAV file."""
        import edge_tts

        audio_chunks = []

        async def _run():
            communicate = edge_tts.Communicate(
                text, self.voice, rate=self.rate, volume=self.volume
            )
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])

        loop = self._get_loop()
        loop.run_until_complete(_run())

        raw_audio = b"".join(audio_chunks)
        pcm = self._decode_mp3_to_pcm(raw_audio, sample_rate)

        if pcm is not None:
            import soundfile as sf
            sf.write(output_path, pcm, sample_rate)
            logger.info(f"TTS WAV saved: {output_path} ({len(pcm)/sample_rate:.1f}s)")
            return output_path
        return ""

    @staticmethod
    def _decode_mp3_to_pcm(mp3_data: bytes, target_sr: int = 16000) -> Optional[np.ndarray]:
        """Decode MP3 bytes to float32 PCM at target sample rate."""
        try:
            import librosa
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
                f.write(mp3_data)
                f.flush()
                audio, sr = librosa.load(f.name, sr=target_sr)
            return audio.astype(np.float32)
        except Exception as e:
            logger.error(f"MP3 decode failed: {e}")
            return None

    def list_voices(self) -> list:
        """List available edge-tts voices."""
        import edge_tts

        async def _run():
            return await edge_tts.list_voices()

        loop = self._get_loop()
        voices = loop.run_until_complete(_run())
        return [
            {"name": v["ShortName"], "gender": v["Gender"], "locale": v["Locale"]}
            for v in voices
        ]

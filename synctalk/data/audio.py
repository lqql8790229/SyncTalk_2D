"""Audio processing utilities for mel spectrogram extraction and feature loading."""

import numpy as np
import torch
import librosa
import librosa.filters
from scipy import signal
from torch.utils.data import Dataset


def load_wav(path: str, sr: int = 16000):
    return librosa.core.load(path, sr=sr)[0]


def preemphasis(wav, k: float = 0.97):
    return signal.lfilter([1, -k], [1], wav)


def melspectrogram(wav, n_fft: int = 800, hop_length: int = 200,
                   win_length: int = 800, sr: int = 16000,
                   n_mels: int = 80, fmin: int = 55, fmax: int = 7600,
                   preemphasis_k: float = 0.97, ref_level_db: int = 20,
                   min_level_db: int = -100, max_abs_value: float = 4.0):
    """Compute normalized mel spectrogram."""
    D = librosa.stft(y=preemphasis(wav, preemphasis_k),
                     n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                     fmin=fmin, fmax=fmax)
    S = np.dot(mel_basis, np.abs(D))
    min_level = np.exp(-5 * np.log(10))
    S = 20 * np.log10(np.maximum(min_level, S)) - ref_level_db
    S = np.clip((2 * max_abs_value) * ((S - min_level_db) / (-min_level_db)) - max_abs_value,
                -max_abs_value, max_abs_value)
    return S


def get_audio_features(features: np.ndarray, index: int, window: int = 8):
    """Extract windowed audio features for a given frame index."""
    left = index - window
    right = index + window
    pad_left = max(0, -left)
    pad_right = max(0, right - features.shape[0])
    left = max(0, left)
    right = min(features.shape[0], right)

    auds = torch.from_numpy(features[left:right])
    if pad_left > 0:
        auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
    if pad_right > 0:
        auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0)
    return auds


class AudDataset(Dataset):
    """Dataset for processing audio into mel-spectrogram frames."""

    def __init__(self, wavpath: str, sr: int = 16000):
        wav = load_wav(wavpath, sr)
        self.orig_mel = melspectrogram(wav).T
        self.data_len = int((self.orig_mel.shape[0] - 16) / 80.0 * 25.0) + 2

    def crop_audio_window(self, spec, start_frame):
        start_idx = int(80.0 * (start_frame / 25.0))
        end_idx = start_idx + 16
        if end_idx > spec.shape[0]:
            end_idx = spec.shape[0]
            start_idx = end_idx - 16
        return spec[start_idx:end_idx, :]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        mel = self.crop_audio_window(self.orig_mel.copy(), idx)
        if mel.shape[0] != 16:
            raise ValueError(f"Expected mel shape[0]=16, got {mel.shape[0]}")
        return torch.FloatTensor(mel.T).unsqueeze(0)

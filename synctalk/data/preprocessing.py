"""Data preprocessing pipeline.

Handles the complete preprocessing of training videos:
1. Extract audio (ffmpeg → WAV 16kHz)
2. Extract frames (convert to 25fps, save as JPGs)
3. Detect facial landmarks (SCRFD + PFLD)
4. Extract audio features (AudioEncoder → .npy)
"""

import os
import logging
from pathlib import Path
from tqdm import tqdm

from ..utils.io import safe_run_command, ensure_dir

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Complete data preprocessing pipeline.

    Args:
        sample_rate: Audio sample rate (default 16000).
        fps: Target video FPS (default 25).
    """

    def __init__(self, sample_rate: int = 16000, fps: int = 25):
        self.sample_rate = sample_rate
        self.fps = fps

    def extract_audio(self, video_path: str, out_path: str):
        """Extract audio from video file."""
        logger.info(f"Extracting audio: {video_path} → {out_path}")
        safe_run_command(
            f"ffmpeg -i {video_path} -f wav -ar {self.sample_rate} {out_path} -y"
        )

    def extract_frames(self, video_path: str):
        """Extract frames from video, converting to target FPS if needed."""
        import cv2
        base_dir = os.path.dirname(video_path)
        full_body_dir = os.path.join(base_dir, "full_body_img")
        ensure_dir(full_body_dir)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        actual_path = video_path
        if fps != self.fps:
            logger.info(f"Converting {fps}fps → {self.fps}fps")
            converted = video_path.replace(".mp4", f"_{self.fps}fps.mp4")
            safe_run_command(
                f'ffmpeg -i {video_path} -vf "fps={self.fps}" '
                f'-c:v libx264 -c:a aac {converted} -y'
            )
            actual_path = converted

        cap = cv2.VideoCapture(actual_path)
        if cap.get(cv2.CAP_PROP_FPS) != self.fps:
            raise ValueError(f"FPS conversion failed. Expected {self.fps}fps.")

        counter = 0
        logger.info("Extracting frames...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(full_body_dir, f"{counter}.jpg"), frame)
            counter += 1
        cap.release()
        logger.info(f"Extracted {counter} frames")
        return counter

    def extract_landmarks(self, video_path: str, device=None):
        """Detect facial landmarks for all frames."""
        from .landmark import LandmarkDetector

        base_dir = os.path.dirname(video_path)
        full_body_dir = os.path.join(base_dir, "full_body_img")
        landmarks_dir = os.path.join(base_dir, "landmarks")
        ensure_dir(landmarks_dir)

        detector = LandmarkDetector(device=device)

        img_files = [f for f in os.listdir(full_body_dir) if f.endswith(".jpg")]
        for img_name in tqdm(img_files, desc="Detecting landmarks"):
            img_path = os.path.join(full_body_dir, img_name)
            lms_path = os.path.join(landmarks_dir, img_name.replace(".jpg", ".lms"))

            try:
                pre_landmark, x1, y1 = detector.detect(img_path)
                with open(lms_path, "w") as f:
                    for p in pre_landmark:
                        f.write(f"{p[0] + x1} {p[1] + y1}\n")
            except ValueError as e:
                logger.warning(f"Skipping {img_name}: {e}")

    def extract_audio_features(self, wav_path: str, device=None):
        """Extract audio features using AudioEncoder."""
        import torch
        import numpy as np
        from torch.utils.data import DataLoader
        from ..models.audio_encoder import AudioEncoder
        from .audio import AudDataset
        from ..utils.device import get_device

        dev = device or get_device()
        model = AudioEncoder().to(dev).eval()
        model.load_pretrained("model/checkpoints/audio_visual_encoder.pth", dev)

        dataset = AudDataset(wav_path)
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

        outputs = []
        with torch.no_grad():
            for mel in data_loader:
                mel = mel.to(dev)
                out = model(mel)
                outputs.append(out)

        outputs = torch.cat(outputs, dim=0).cpu()
        first_frame = outputs[:1]
        last_frame = outputs[-1:]
        outputs = torch.cat([first_frame, outputs, last_frame], dim=0)

        save_path = wav_path.replace(".wav", "_ave.npy")
        np.save(save_path, outputs.numpy())
        logger.info(f"Audio features saved: {save_path} shape={outputs.shape}")
        return save_path

    def process(self, video_path: str, device=None):
        """Run complete preprocessing pipeline."""
        base_dir = os.path.dirname(video_path)
        wav_path = os.path.join(base_dir, "aud.wav")
        ensure_dir(os.path.join(base_dir, "landmarks"))

        logger.info(f"=== Processing: {video_path} ===")

        self.extract_audio(video_path, wav_path)
        self.extract_frames(video_path)
        self.extract_landmarks(video_path, device=device)
        self.extract_audio_features(wav_path, device=device)

        logger.info("=== Preprocessing complete ===")

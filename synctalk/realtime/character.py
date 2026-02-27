"""Character management for real-time lip-sync.

A 'Character' encapsulates all the pre-processed data needed for
real-time inference: original frames, landmarks, and trained model.
"""

import os
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from ..models.unet import UNet
from ..configs.base import ModelConfig
from ..utils.io import read_landmarks, get_crop_region

logger = logging.getLogger(__name__)


class Character:
    """Pre-loaded character data for real-time rendering.

    Args:
        name: Character/dataset name.
        dataset_dir: Path to dataset directory (default: ./dataset/{name}).
        checkpoint_dir: Path to checkpoint directory (default: ./checkpoint/{name}).
        config: ModelConfig (default: 328px AVE).
        device: Torch device.
    """

    def __init__(self, name: str, dataset_dir: str = None,
                 checkpoint_dir: str = None, config: ModelConfig = None,
                 device: torch.device = None):
        self.name = name
        self.config = config or ModelConfig(resolution=328, asr_mode="ave")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset_dir = dataset_dir or f"./dataset/{name}"
        self.checkpoint_dir = checkpoint_dir or f"./checkpoint/{name}"

        self._load_model()
        self._load_frames()

    def _load_model(self):
        """Load the trained UNet model."""
        ckpt_dir = Path(self.checkpoint_dir)
        ckpt_files = sorted(ckpt_dir.glob("*.pth"),
                            key=lambda p: p.stem if not p.stem.isdigit() else int(p.stem))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoints in {self.checkpoint_dir}")

        best_ckpt = ckpt_dir / "best.pth"
        checkpoint_path = str(best_ckpt if best_ckpt.exists() else ckpt_files[-1])

        self.model = UNet.from_config(self.config).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.load_state_dict(ckpt)
        self.model.eval()

        self.use_fp16 = self.device.type == "cuda"
        if self.use_fp16:
            self.model = self.model.half()
            logger.info("FP16 enabled for GPU inference")

        logger.info(f"Loaded model: {checkpoint_path}")

    def _load_frames(self):
        """Pre-load all frames and landmarks into memory."""
        img_dir = os.path.join(self.dataset_dir, "full_body_img")
        lms_dir = os.path.join(self.dataset_dir, "landmarks")

        n_frames = len([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
        if n_frames == 0:
            raise FileNotFoundError(f"No frames in {img_dir}")

        self.frames = []
        self.landmarks = []
        self.crop_regions = []

        for i in range(n_frames):
            img = cv2.imread(os.path.join(img_dir, f"{i}.jpg"))
            lms = read_landmarks(os.path.join(lms_dir, f"{i}.lms"))
            region = get_crop_region(lms)
            self.frames.append(img)
            self.landmarks.append(lms)
            self.crop_regions.append(region)

        self.n_frames = n_frames
        sample = self.frames[0]
        self.frame_h, self.frame_w = sample.shape[:2]
        logger.info(f"Loaded {n_frames} frames ({self.frame_w}x{self.frame_h})")

    def get_frame(self, idx: int):
        """Get frame data by index (with wrapping)."""
        idx = idx % self.n_frames
        return self.frames[idx].copy(), self.landmarks[idx], self.crop_regions[idx]

    @torch.no_grad()
    def render_frame(self, frame_idx: int, audio_features: np.ndarray) -> np.ndarray:
        """Render a single lip-synced frame.

        Args:
            frame_idx: Source frame index.
            audio_features: Windowed audio features [16, 512].

        Returns:
            Rendered BGR frame (original resolution).
        """
        cfg = self.config
        crop_size = cfg.crop_size
        c_start, c_end = cfg.inner_crop
        mx, my, mw, mh = cfg.mask_rect
        feat_shape = cfg.audio_feat_shape

        img, lms, (xmin, ymin, xmax, ymax, width) = self.get_frame(frame_idx)

        crop_img = img[ymin:ymax, xmin:xmax]
        if crop_img.size == 0:
            return img
        h, w = crop_img.shape[:2]
        crop_img = cv2.resize(crop_img, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
        crop_img_ori = crop_img.copy()

        img_real_ex = crop_img[c_start:c_end, c_start:c_end].copy()
        img_real_ex_ori = img_real_ex.copy()
        img_masked = cv2.rectangle(img_real_ex_ori, (mx, my, mw, mh), (0, 0, 0), -1)

        img_masked_t = torch.from_numpy(
            img_masked.transpose(2, 0, 1).astype(np.float32) / 255.0)
        img_real_t = torch.from_numpy(
            img_real_ex.transpose(2, 0, 1).astype(np.float32) / 255.0)
        img_concat = torch.cat([img_real_t, img_masked_t], dim=0).unsqueeze(0).to(self.device)

        audio_feat = torch.from_numpy(audio_features).reshape(*feat_shape)
        audio_feat = audio_feat.unsqueeze(0).to(self.device)

        if self.use_fp16:
            img_concat = img_concat.half()
            audio_feat = audio_feat.half()

        pred = self.model(img_concat, audio_feat)[0]
        pred = pred.float().cpu().numpy().transpose(1, 2, 0) * 255
        pred = np.array(pred, dtype=np.uint8)

        crop_img_ori[c_start:c_end, c_start:c_end] = pred
        crop_img_ori = cv2.resize(crop_img_ori, (w, h), interpolation=cv2.INTER_CUBIC)
        img[ymin:ymax, xmin:xmax] = crop_img_ori

        return img

    def get_silent_frame(self, frame_idx: int) -> np.ndarray:
        """Return the original frame without lip-sync (for silent segments)."""
        img, _, _ = self.get_frame(frame_idx)
        return img

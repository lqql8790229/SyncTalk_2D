"""Unified inference engine with batch processing and device abstraction.

Merges inference.py and inference_328.py into a single engine.
"""

import os
import logging

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..configs.base import SyncTalkConfig, ModelConfig
from ..models.unet import UNet
from ..models.audio_encoder import AudioEncoder
from ..data.audio import AudDataset, get_audio_features
from ..utils.device import get_device
from ..utils.io import safe_run_command, safe_remove, ensure_dir, read_landmarks, get_crop_region

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Lip-sync video inference engine.

    Args:
        config: SyncTalkConfig.
        device_str: Device override ("auto", "cuda", "cpu").
    """

    def __init__(self, config: SyncTalkConfig, device_str: str = "auto"):
        self.config = config
        self.device = get_device(device_str)
        self.model_config = config.model

    def load_model(self, checkpoint_path: str) -> UNet:
        """Load trained UNet model."""
        net = UNet.from_config(self.model_config).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if "model_state_dict" in ckpt:
            net.load_state_dict(ckpt["model_state_dict"])
        else:
            net.load_state_dict(ckpt)
        net.eval()
        logger.info(f"Loaded model: {checkpoint_path}")
        return net

    def extract_audio_features(self, audio_path: str) -> np.ndarray:
        """Extract audio features from WAV file."""
        model = AudioEncoder().to(self.device).eval()
        model.load_pretrained("model/checkpoints/audio_visual_encoder.pth", self.device)

        dataset = AudDataset(audio_path)
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

        outputs = []
        for mel in data_loader:
            mel = mel.to(self.device)
            with torch.no_grad():
                out = model(mel)
            outputs.append(out)

        outputs = torch.cat(outputs, dim=0).cpu()
        first_frame, last_frame = outputs[:1], outputs[-1:]
        audio_feats = torch.cat(
            [first_frame.repeat(1, 1), outputs, last_frame.repeat(1, 1)], dim=0
        ).numpy()
        logger.info(f"Audio features: shape={audio_feats.shape}")
        return audio_feats

    def generate(self, name: str, audio_path: str, checkpoint_path: str = None,
                 output_path: str = None, start_frame: int = 0,
                 use_parsing: bool = False):
        """Generate lip-sync video.

        Args:
            name: Dataset name (expects data in ./dataset/{name}/).
            audio_path: Path to audio WAV file.
            checkpoint_path: Path to model checkpoint. Auto-detects if None.
            output_path: Output video path. Auto-generates if None.
            start_frame: Starting frame index.
            use_parsing: Whether to use semantic parsing masks.
        """
        cfg = self.model_config
        mode = cfg.asr_mode
        crop_size = cfg.crop_size
        c_start, c_end = cfg.inner_crop

        if checkpoint_path is None:
            ckpt_dir = os.path.join("./checkpoint", name)
            ckpt_files = sorted(os.listdir(ckpt_dir), key=lambda x: int(x.split(".")[0]))
            checkpoint_path = os.path.join(ckpt_dir, ckpt_files[-1])

        if output_path is None:
            ensure_dir("./result")
            ckpt_name = os.path.basename(checkpoint_path).split(".")[0]
            audio_name = os.path.basename(audio_path).split(".")[0]
            output_path = f"./result/{name}_{audio_name}_{ckpt_name}.mp4"

        net = self.load_model(checkpoint_path)
        audio_feats = self.extract_audio_features(audio_path)

        dataset_dir = os.path.join("./dataset", name)
        img_dir = os.path.join(dataset_dir, "full_body_img/")
        lms_dir = os.path.join(dataset_dir, "landmarks/")
        len_img = len(os.listdir(img_dir)) - 1
        exm_img = cv2.imread(os.path.join(img_dir, "0.jpg"))
        if exm_img is None:
            raise FileNotFoundError(f"Cannot read {img_dir}0.jpg")
        orig_h, orig_w = exm_img.shape[:2]

        parsing_dir = os.path.join(dataset_dir, "parsing/") if use_parsing else None

        fps = self.config.inference.fps
        if mode == "wenet":
            fps = 20

        temp_path = output_path.replace(".mp4", "_temp.mp4")
        video_writer = cv2.VideoWriter(
            temp_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            fps, (orig_w, orig_h),
        )

        step_stride = 0
        img_idx = 0

        for i in tqdm(range(audio_feats.shape[0]), desc="Generating"):
            if img_idx > len_img - 1:
                step_stride = -1
            if img_idx < 1:
                step_stride = 1
            img_idx += step_stride

            frame_idx = img_idx + start_frame
            img = cv2.imread(os.path.join(img_dir, f"{frame_idx}.jpg"))
            lms = read_landmarks(os.path.join(lms_dir, f"{frame_idx}.lms"))
            xmin, ymin, xmax, ymax, width = get_crop_region(lms)

            if use_parsing and parsing_dir:
                parsing = cv2.imread(os.path.join(parsing_dir, f"{frame_idx}.png"))

            crop_img = img[ymin:ymax, xmin:xmax]
            crop_img_par = crop_img.copy() if use_parsing else None
            h, w = crop_img.shape[:2]
            crop_img = cv2.resize(crop_img, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
            crop_img_ori = crop_img.copy()

            img_real_ex = crop_img[c_start:c_end, c_start:c_end].copy()
            img_real_ex_ori = img_real_ex.copy()
            mx, my, mw, mh = cfg.mask_rect
            img_masked = cv2.rectangle(img_real_ex_ori, (mx, my, mw, mh), (0, 0, 0), -1)

            img_masked = img_masked.transpose(2, 0, 1).astype(np.float32)
            img_real_ex = img_real_ex.transpose(2, 0, 1).astype(np.float32)

            img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
            img_masked_T = torch.from_numpy(img_masked / 255.0)
            img_concat_T = torch.cat([img_real_ex_T, img_masked_T], dim=0).unsqueeze(0)

            audio_feat = get_audio_features(audio_feats, i)
            feat_shape = cfg.audio_feat_shape
            audio_feat = audio_feat.reshape(*feat_shape).unsqueeze(0)

            img_concat_T = img_concat_T.to(self.device)
            audio_feat = audio_feat.to(self.device)

            with torch.no_grad():
                pred = net(img_concat_T, audio_feat)[0]

            pred = pred.cpu().numpy().transpose(1, 2, 0) * 255
            pred = np.array(pred, dtype=np.uint8)

            crop_img_ori[c_start:c_end, c_start:c_end] = pred
            crop_img_ori = cv2.resize(crop_img_ori, (w, h), interpolation=cv2.INTER_CUBIC)

            if use_parsing and parsing_dir:
                crop_parsing = parsing[ymin:ymax, xmin:xmax]
                mask = ((crop_parsing == [0, 0, 255]).all(axis=2) |
                        (crop_parsing == [255, 255, 255]).all(axis=2))
                crop_img_ori[mask] = crop_img_par[mask]

            img[ymin:ymax, xmin:xmax] = crop_img_ori
            video_writer.write(img)

        video_writer.release()

        crf = self.config.inference.output_crf
        codec = self.config.inference.output_codec
        safe_run_command(
            f"ffmpeg -i {temp_path} -i {audio_path} "
            f"-c:v {codec} -c:a aac -crf {crf} {output_path} -y"
        )
        safe_remove(temp_path)
        logger.info(f"Saved: {output_path}")
        return output_path

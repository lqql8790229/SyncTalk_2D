"""Unified dataset classes for training.

Merges datasetsss.py (160px) and datasetsss_328.py (328px) into
parameterized datasets.
"""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..configs.base import ModelConfig
from .audio import get_audio_features


class LipSyncDataset(Dataset):
    """Main training dataset for UNet lip-sync model.

    Args:
        img_dir: Path to dataset directory.
        config: ModelConfig with resolution settings.
    """

    def __init__(self, img_dir: str, config: ModelConfig):
        self.config = config
        self.mode = config.asr_mode
        self.crop_size = config.crop_size
        self.inner_crop = config.inner_crop
        self.mask_rect = config.mask_rect

        self.img_path_list = []
        self.lms_path_list = []

        full_body_dir = os.path.join(img_dir, "full_body_img")
        landmarks_dir = os.path.join(img_dir, "landmarks")

        for i in range(len(os.listdir(full_body_dir))):
            self.img_path_list.append(os.path.join(full_body_dir, f"{i}.jpg"))
            self.lms_path_list.append(os.path.join(landmarks_dir, f"{i}.lms"))

        audio_map = {
            "wenet": "aud_wenet.npy",
            "hubert": "aud_hu.npy",
            "ave": "aud_ave.npy",
        }
        audio_feats_path = os.path.join(img_dir, audio_map[self.mode])
        self.audio_feats = np.load(audio_feats_path).astype(np.float32)

    def __len__(self):
        return self.audio_feats.shape[0] - 1

    def _read_landmarks(self, lms_path):
        lms_list = []
        with open(lms_path, "r") as f:
            for line in f.read().splitlines():
                arr = np.array(line.split(" "), dtype=np.float32)
                lms_list.append(arr)
        return np.array(lms_list, dtype=np.int32)

    def _get_crop_region(self, lms):
        xmin = lms[1][0]
        ymin = lms[52][1]
        xmax = lms[31][0]
        width = xmax - xmin
        ymax = ymin + width
        return xmin, ymin, xmax, ymax

    def process_img(self, img, lms_path, img_ex, lms_path_ex):
        lms = self._read_landmarks(lms_path)
        xmin, ymin, xmax, ymax = self._get_crop_region(lms)

        c_start, c_end = self.inner_crop
        crop_img = img[ymin:ymax, xmin:xmax]
        crop_img = cv2.resize(crop_img, (self.crop_size, self.crop_size), cv2.INTER_AREA)
        img_real = crop_img[c_start:c_end, c_start:c_end].copy()
        img_real_ori = img_real.copy()
        mx, my, mw, mh = self.mask_rect
        img_masked = cv2.rectangle(img_real, (mx, my, mw, mh), (0, 0, 0), -1)

        lms_ex = self._read_landmarks(lms_path_ex)
        xmin_ex, ymin_ex, xmax_ex, ymax_ex = self._get_crop_region(lms_ex)
        crop_img_ex = img_ex[ymin_ex:ymax_ex, xmin_ex:xmax_ex]
        crop_img_ex = cv2.resize(crop_img_ex, (self.crop_size, self.crop_size), cv2.INTER_AREA)
        img_real_ex = crop_img_ex[c_start:c_end, c_start:c_end].copy()

        img_real_ori = torch.from_numpy(
            img_real_ori.transpose(2, 0, 1).astype(np.float32) / 255.0)
        img_masked = torch.from_numpy(
            img_masked.transpose(2, 0, 1).astype(np.float32) / 255.0)
        img_real_ex = torch.from_numpy(
            img_real_ex.transpose(2, 0, 1).astype(np.float32) / 255.0)

        img_concat = torch.cat([img_real_ex, img_masked], dim=0)
        return img_concat, img_real_ori

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path_list[idx])
        lms_path = self.lms_path_list[idx]

        ex_int = random.randint(0, len(self) - 1)
        img_ex = cv2.imread(self.img_path_list[ex_int])
        lms_path_ex = self.lms_path_list[ex_int]

        img_concat, img_real = self.process_img(img, lms_path, img_ex, lms_path_ex)

        audio_feat = get_audio_features(self.audio_feats, idx)
        feat_shape = self.config.audio_feat_shape
        audio_feat = audio_feat.reshape(*feat_shape)

        return img_concat, img_real, audio_feat


class SyncNetDataset:
    """Dataset for SyncNet training.

    Args:
        dataset_dir: Path to dataset directory.
        config: ModelConfig with resolution settings.
    """

    def __init__(self, dataset_dir: str, config: ModelConfig):
        self.config = config
        self.mode = config.asr_mode
        self.crop_size = config.crop_size
        self.inner_crop = config.inner_crop

        self.img_path_list = []
        self.lms_path_list = []

        full_body_dir = os.path.join(dataset_dir, "full_body_img")
        for i in range(len(os.listdir(full_body_dir))):
            self.img_path_list.append(os.path.join(full_body_dir, f"{i}.jpg"))
            self.lms_path_list.append(
                os.path.join(dataset_dir, "landmarks", f"{i}.lms"))

        audio_map = {"wenet": "aud_wenet.npy", "hubert": "aud_hu.npy", "ave": "aud_ave.npy"}
        audio_path = os.path.join(dataset_dir, audio_map[self.mode])
        self.audio_feats = np.load(audio_path).astype(np.float32)

    def __len__(self):
        return self.audio_feats.shape[0] - 1

    def _read_landmarks(self, lms_path):
        lms_list = []
        with open(lms_path, "r") as f:
            for line in f.read().splitlines():
                arr = np.array(line.split(" "), dtype=np.float32)
                lms_list.append(arr)
        return np.array(lms_list, dtype=np.int32)

    def process_img(self, img, lms_path, img_ex, lms_path_ex):
        lms = self._read_landmarks(lms_path)
        xmin, ymin = lms[1][0], lms[52][1]
        xmax = lms[31][0]
        width = xmax - xmin
        ymax = ymin + width

        c_start, c_end = self.inner_crop
        crop_img = img[ymin:ymax, xmin:xmax]
        crop_img = cv2.resize(crop_img, (self.crop_size, self.crop_size), cv2.INTER_AREA)
        img_real = crop_img[c_start:c_end, c_start:c_end].copy()
        img_real_ori = img_real.transpose(2, 0, 1).astype(np.float32)
        return torch.from_numpy(img_real_ori / 255.0)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path_list[idx])
        lms_path = self.lms_path_list[idx]

        ex_int = random.randint(0, len(self) - 1)
        img_ex = cv2.imread(self.img_path_list[ex_int])
        lms_path_ex = self.lms_path_list[ex_int]

        img_real = self.process_img(img, lms_path, img_ex, lms_path_ex)

        audio_feat = get_audio_features(self.audio_feats, idx)
        feat_shape = self.config.audio_feat_shape
        audio_feat = audio_feat.reshape(*feat_shape)

        y = torch.ones(1).float()
        return img_real, audio_feat, y

"""Facial landmark detection using PFLD-MobileOne."""

import math
import cv2
import numpy as np
import torch
import torchvision
from pathlib import Path

from .face_detection import FaceDetector


class LandmarkDetector:
    """PFLD-based facial landmark detector.

    Args:
        pfld_checkpoint: Path to PFLD checkpoint.
        mean_face_path: Path to mean face txt file.
        scrfd_model_path: Path to SCRFD ONNX model.
        device: Torch device to use.
    """

    def __init__(self, pfld_checkpoint: str = "data_utils/checkpoint_epoch_335.pth.tar",
                 mean_face_path: str = "data_utils/mean_face.txt",
                 scrfd_model_path: str = "data_utils/scrfd_2.5g_kps.onnx",
                 device: torch.device = None):
        import sys
        sys.path.insert(0, str(Path("data_utils").resolve()))
        from pfld_mobileone import PFLD_GhostOne as PFLDInference
        sys.path.pop(0)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(mean_face_path, 'r') as f:
            self.mean_face = np.asarray(f.read().split(' '), dtype=np.float32)

        self.det_net = FaceDetector(scrfd_model_path, conf_threshold=0.1, nms_threshold=0.5)

        checkpoint = torch.load(pfld_checkpoint, map_location=self.device)
        self.pfld_backbone = PFLDInference().to(self.device)
        self.pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
        self.pfld_backbone.eval()

    def _detect_face(self, img):
        """Detect and crop the first face in the image."""
        bboxes, indices, kps = self.det_net.detect(img)

        for i in indices:
            x1 = int(bboxes[i, 0])
            y1 = int(bboxes[i, 1])
            x2 = int(bboxes[i, 0] + bboxes[i, 2])
            y2 = int(bboxes[i, 1] + bboxes[i, 3])

            w, h = x2 - x1, y2 - y1
            cx, cy = (x2 + x1) // 2, (y2 + y1) // 2
            size = int(max(w, h) * 1.05)

            crop_x1 = cx - size // 2
            crop_y1 = cy - size // 2
            crop_x2 = crop_x1 + size
            crop_y2 = crop_y1 + size

            img_h, img_w = img.shape[:2]
            dx = max(0, -crop_x1)
            dy = max(0, -crop_y1)
            edx = max(0, crop_x2 - img_w)
            edy = max(0, crop_y2 - img_h)

            crop_x1, crop_y1 = max(0, crop_x1), max(0, crop_y1)
            crop_x2, crop_y2 = min(img_w, crop_x2), min(img_h, crop_y2)

            cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
            if dx > 0 or dy > 0 or edx > 0 or edy > 0:
                cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx,
                                             cv2.BORDER_CONSTANT, 0)
                crop_y1 -= dy
                crop_x1 -= dx

            return cropped, [crop_x1, crop_y1, crop_x2, crop_y2]

        raise ValueError("No face detected in image")

    def detect(self, img_path: str):
        """Detect landmarks for the primary face.

        Returns:
            landmarks: [N, 2] array of landmark coordinates (in original image space)
            x1, y1: crop region offset
        """
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        cropped, box = self._detect_face(img)
        x1, y1 = box[0], box[1]
        h, w = cropped.shape[:2]

        inp = cv2.resize(cropped, (192, 192))
        inp = inp.astype(np.float32) / 255.0
        inp = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            landmarks = self.pfld_backbone(inp)

        pre_landmark = landmarks[0].cpu().numpy() + self.mean_face
        pre_landmark = pre_landmark.reshape(-1, 2)
        pre_landmark[:, 0] *= w
        pre_landmark[:, 1] *= h

        return pre_landmark.astype(np.int32), x1, y1

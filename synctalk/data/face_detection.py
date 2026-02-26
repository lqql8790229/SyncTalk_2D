"""Face detection using SCRFD ONNX model."""

import cv2
import numpy as np
from pathlib import Path


class FaceDetector:
    """SCRFD-based face detector.

    Args:
        model_path: Path to SCRFD ONNX model.
        conf_threshold: Confidence threshold.
        nms_threshold: NMS threshold.
    """

    def __init__(self, model_path: str = "data_utils/scrfd_2.5g_kps.onnx",
                 conf_threshold: float = 0.5, nms_threshold: float = 0.5):
        self.inp_width = 640
        self.inp_height = 640
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.net = cv2.dnn.readNet(str(model_path))
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2

    def _resize_image(self, srcimg):
        padh, padw = 0, 0
        newh, neww = self.inp_height, self.inp_width

        if srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                neww = int(self.inp_width / hw_scale)
                img = cv2.resize(srcimg, (neww, self.inp_height), interpolation=cv2.INTER_AREA)
                padw = int((self.inp_width - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, padw, self.inp_width - neww - padw,
                                         cv2.BORDER_CONSTANT, value=0)
            else:
                newh = int(self.inp_height * hw_scale) + 1
                img = cv2.resize(srcimg, (self.inp_width, newh), interpolation=cv2.INTER_AREA)
                padh = int((self.inp_height - newh) * 0.5)
                img = cv2.copyMakeBorder(img, padh, self.inp_height - newh - padh, 0, 0,
                                         cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, (self.inp_width, self.inp_height), interpolation=cv2.INTER_AREA)

        return img, newh, neww, padh, padw

    def detect(self, srcimg):
        """Detect faces in image.

        Returns:
            bboxes: [N, 4] array (x, y, w, h)
            indices: list of valid detection indices
            kpss: [N, 5, 2] keypoint array
        """
        img, newh, neww, padh, padw = self._resize_image(srcimg)
        blob = cv2.dnn.blobFromImage(img, 1.0 / 128,
                                      (self.inp_width, self.inp_height),
                                      (127.5, 127.5, 127.5), swapRB=True)
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        scores_list, bboxes_list, kpss_list = [], [], []
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outs[idx][0]
            bbox_preds = outs[idx + self.fmc][0] * stride
            kps_preds = outs[idx + self.fmc * 2][0] * stride

            height = blob.shape[2] // stride
            width = blob.shape[3] // stride
            anchor_centers = np.stack(
                np.mgrid[:height, :width][::-1], axis=-1
            ).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if self._num_anchors > 1:
                anchor_centers = np.stack(
                    [anchor_centers] * self._num_anchors, axis=1
                ).reshape((-1, 2))

            pos_inds = np.where(scores >= self.conf_threshold)[0]

            x1 = anchor_centers[:, 0] - bbox_preds[:, 0]
            y1 = anchor_centers[:, 1] - bbox_preds[:, 1]
            x2 = anchor_centers[:, 0] + bbox_preds[:, 2]
            y2 = anchor_centers[:, 1] + bbox_preds[:, 3]
            bboxes = np.stack([x1, y1, x2, y2], axis=-1)

            preds = []
            for i in range(0, kps_preds.shape[1], 2):
                px = anchor_centers[:, i % 2] + kps_preds[:, i]
                py = anchor_centers[:, i % 2 + 1] + kps_preds[:, i + 1]
                preds.extend([px, py])
            kpss = np.stack(preds, axis=-1).reshape((kps_preds.shape[0], -1, 2))

            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            kpss_list.append(kpss[pos_inds])

        scores = np.vstack(scores_list).ravel()
        bboxes = np.vstack(bboxes_list)
        kpss = np.vstack(kpss_list)

        bboxes[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]
        ratioh = srcimg.shape[0] / newh
        ratiow = srcimg.shape[1] / neww
        bboxes[:, 0] = (bboxes[:, 0] - padw) * ratiow
        bboxes[:, 1] = (bboxes[:, 1] - padh) * ratioh
        bboxes[:, 2] *= ratiow
        bboxes[:, 3] *= ratioh
        kpss[:, :, 0] = (kpss[:, :, 0] - padw) * ratiow
        kpss[:, :, 1] = (kpss[:, :, 1] - padh) * ratioh

        indices = cv2.dnn.NMSBoxes(
            bboxes.tolist(), scores.tolist(),
            self.conf_threshold, self.nms_threshold
        )
        return bboxes, indices, kpss

"""CCPD-master/rpnet/demo.py 推理（逐行对应）。"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

from app.ccpd.charset import ads, alphabets, provinces
from app.ccpd.load_data import preprocess_image_bgr
from app.ccpd.model import fh02, numClasses, numPoints

logger = logging.getLogger(__name__)

imgSize = (480, 480)


def _remap_state_dict_cpu(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """CPU 无 DataParallel 时，将官方 checkpoint 映射到 fh02 结构。"""
    mapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module.wR2.module."):
            new_key = "wR2." + new_key[len("module.wR2.module.") :]
        elif new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        mapped[new_key] = value
    return mapped


def _build_model(model_path: str, device: torch.device) -> nn.Module:
    """demo.py lines 254-258。"""
    model_conv = fh02(numPoints, numClasses)
    state = torch.load(model_path, map_location=device, weights_only=False)

    if device.type == "cuda":
        model_conv = nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
        model_conv.load_state_dict(state)
    else:
        # CPU: 扁平化加载，forward 走 _wr2_module()
        flat = fh02(numPoints, numClasses)
        flat.load_state_dict(_remap_state_dict_cpu(state), strict=True)
        model_conv = flat

    model_conv.to(device)
    model_conv.eval()
    return model_conv


def _demo_forward(model: nn.Module, image_bgr: np.ndarray, device: torch.device) -> dict[str, Any]:
    """demo.py lines 265-286。"""
    img_h, img_w = image_bgr.shape[:2]
    tensor = torch.from_numpy(preprocess_image_bgr(image_bgr, imgSize)).unsqueeze(0).to(device)

    with torch.no_grad():
        fps_pred, y_pred = model(tensor)

    output_y = [el.data.cpu().numpy().tolist() for el in y_pred]
    label_pred = [t[0].index(max(t[0])) for t in output_y]

    cx, cy, w, h = fps_pred.data.cpu().numpy()[0].tolist()
    left_up = [(cx - w / 2) * img_w, (cy - h / 2) * img_h]
    right_down = [(cx + w / 2) * img_w, (cy + h / 2) * img_h]
    bbox = [int(left_up[0]), int(left_up[1]), int(right_down[0]), int(right_down[1])]

    plate_number = (
        provinces[label_pred[0]]
        + alphabets[label_pred[1]]
        + ads[label_pred[2]]
        + ads[label_pred[3]]
        + ads[label_pred[4]]
        + ads[label_pred[5]]
        + ads[label_pred[6]]
    )
    lpn = (
        alphabets[label_pred[1]]
        + ads[label_pred[2]]
        + ads[label_pred[3]]
        + ads[label_pred[4]]
        + ads[label_pred[5]]
        + ads[label_pred[6]]
    )

    confidences = [max(t[0]) for t in output_y if t and t[0]]
    confidence = min(1.0, sum(confidences) / max(len(confidences), 1) / 100.0)

    return {
        "plate_number": plate_number,
        "lpn": lpn,
        "bbox": bbox,
        "indices": label_pred,
        "left_up": left_up,
        "right_down": right_down,
        "confidence": confidence,
    }


class CCPDRPNetRecognizer:
    def __init__(self, model_path: str):
        self._model_path = model_path
        self._model: nn.Module | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            self._model = _build_model(self._model_path, self._device)
            logger.info("CCPD RPNet loaded from %s on %s", self._model_path, self._device)
        return self._model

    def recognize(self, image_bgr: np.ndarray) -> dict[str, Any]:
        if image_bgr is None or image_bgr.size == 0:
            return {"plate_number": "", "lpn": "", "bbox": [0, 0, 0, 0], "indices": []}
        return _demo_forward(self.model, image_bgr, self._device)

    def recognize_path(self, img_path: str) -> dict[str, Any]:
        """demo.py: img = cv2.imread(ims[0])"""
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise ValueError(f"无法读取图像: {img_path}")
        return self.recognize(image_bgr)

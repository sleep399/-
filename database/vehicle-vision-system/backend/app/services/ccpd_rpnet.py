"""CCPD RPNet end-to-end license plate detection and recognition.

Robust inference: multi-variant preprocessing, best-pass selection, crop refinement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.utils.helpers import (
    detect_plate_color,
    detect_plate_regions,
    indices_to_plate,
    refine_one_vs_i,
    select_plate_bbox,
)

logger = logging.getLogger(__name__)

IMG_SIZE = (480, 480)
PROV_NUM, ALPHA_NUM, AD_NUM = 38, 25, 35
CCPD_ASPECT = 1160 / 720

_DIGIT_PREF = {
    frozenset({4, 29}): 29,   # E / 5
    frozenset({1, 25}): 25,   # B / 1
    frozenset({0, 24}): 24,   # A / 0
    frozenset({8, 32}): 32,   # H / 8
    frozenset({11, 31}): 31,  # K / 7
}


def _roi_pooling_ims(features: torch.Tensor, rois: torch.Tensor, size: tuple[int, int] = (16, 8)) -> torch.Tensor:
    out_h, out_w = size[1], size[0]
    pooled: list[torch.Tensor] = []
    _, _, fh, fw = features.shape
    for i in range(features.size(0)):
        x1, y1, x2, y2 = rois[i].long()
        x1 = int(x1.clamp(0, fw - 1).item())
        y1 = int(y1.clamp(0, fh - 1).item())
        x2 = int(x2.clamp(x1 + 1, fw - 1).item())
        y2 = int(y2.clamp(y1 + 1, fh - 1).item())
        crop = features[i : i + 1, :, y1 : y2 + 1, x1 : x2 + 1]
        pooled.append(F.adaptive_max_pool2d(crop, (out_h, out_w)))
    return torch.cat(pooled, dim=0)


class WR2(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        hidden1 = nn.Sequential(
            nn.Conv2d(3, 48, 5, padding=2, stride=2), nn.BatchNorm2d(48), nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1), nn.Dropout(0.2),
        )
        hidden2 = nn.Sequential(
            nn.Conv2d(48, 64, 5, padding=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, stride=1, padding=1), nn.Dropout(0.2),
        )
        hidden3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, padding=2), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1), nn.Dropout(0.2),
        )
        hidden4 = nn.Sequential(
            nn.Conv2d(128, 160, 5, padding=2), nn.BatchNorm2d(160), nn.ReLU(),
            nn.MaxPool2d(2, stride=1, padding=1), nn.Dropout(0.2),
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(160, 192, 5, padding=2), nn.BatchNorm2d(192), nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1), nn.Dropout(0.2),
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(192, 192, 5, padding=2), nn.BatchNorm2d(192), nn.ReLU(),
            nn.MaxPool2d(2, stride=1, padding=1), nn.Dropout(0.2),
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(192, 192, 5, padding=2), nn.BatchNorm2d(192), nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1), nn.Dropout(0.2),
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(192, 192, 5, padding=2), nn.BatchNorm2d(192), nn.ReLU(),
            nn.MaxPool2d(2, stride=1, padding=1), nn.Dropout(0.2),
        )
        hidden9 = nn.Sequential(
            nn.Conv2d(192, 192, 3, padding=1), nn.BatchNorm2d(192), nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1), nn.Dropout(0.2),
        )
        hidden10 = nn.Sequential(
            nn.Conv2d(192, 192, 3, padding=1), nn.BatchNorm2d(192), nn.ReLU(),
            nn.MaxPool2d(2, stride=1, padding=1), nn.Dropout(0.2),
        )
        self.features = nn.Sequential(
            hidden1, hidden2, hidden3, hidden4, hidden5,
            hidden6, hidden7, hidden8, hidden9, hidden10,
        )
        self.classifier = nn.Sequential(
            nn.Linear(23232, 100), nn.Linear(100, 100), nn.Linear(100, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.features(x)
        return self.classifier(x1.view(x1.size(0), -1))


class FH02(nn.Module):
    def __init__(self):
        super().__init__()
        self.wR2 = WR2(4)
        self.classifier1 = nn.Sequential(nn.Linear(53248, 128), nn.Linear(128, PROV_NUM))
        self.classifier2 = nn.Sequential(nn.Linear(53248, 128), nn.Linear(128, ALPHA_NUM))
        self.classifier3 = nn.Sequential(nn.Linear(53248, 128), nn.Linear(128, AD_NUM))
        self.classifier4 = nn.Sequential(nn.Linear(53248, 128), nn.Linear(128, AD_NUM))
        self.classifier5 = nn.Sequential(nn.Linear(53248, 128), nn.Linear(128, AD_NUM))
        self.classifier6 = nn.Sequential(nn.Linear(53248, 128), nn.Linear(128, AD_NUM))
        self.classifier7 = nn.Sequential(nn.Linear(53248, 128), nn.Linear(128, AD_NUM))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x0 = self.wR2.features[0](x)
        _x1 = self.wR2.features[1](x0)
        x2 = self.wR2.features[2](_x1)
        _x3 = self.wR2.features[3](x2)
        x4 = self.wR2.features[4](_x3)
        _x5 = self.wR2.features[5](x4)
        x6 = self.wR2.features[6](_x5)
        x7 = self.wR2.features[7](x6)
        x8 = self.wR2.features[8](x7)
        x9 = self.wR2.features[9](x8)
        box_loc = self.wR2.classifier(x9.view(x9.size(0), -1))

        device, dtype = x.device, x.dtype
        h1, w1 = _x1.shape[2], _x1.shape[3]
        p1 = torch.tensor([[w1, 0, 0, 0], [0, h1, 0, 0], [0, 0, w1, 0], [0, 0, 0, h1]], device=device, dtype=dtype)
        h2, w2 = _x3.shape[2], _x3.shape[3]
        p2 = torch.tensor([[w2, 0, 0, 0], [0, h2, 0, 0], [0, 0, w2, 0], [0, 0, 0, h2]], device=device, dtype=dtype)
        h3, w3 = _x5.shape[2], _x5.shape[3]
        p3 = torch.tensor([[w3, 0, 0, 0], [0, h3, 0, 0], [0, 0, w3, 0], [0, 0, 0, h3]], device=device, dtype=dtype)
        postfix = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
            device=device, dtype=dtype,
        )
        box_new = box_loc.mm(postfix).clamp(min=0, max=1)

        roi1 = _roi_pooling_ims(_x1, box_new.mm(p1), size=(16, 8))
        roi2 = _roi_pooling_ims(_x3, box_new.mm(p2), size=(16, 8))
        roi3 = _roi_pooling_ims(_x5, box_new.mm(p3), size=(16, 8))
        rois = torch.cat((roi1, roi2, roi3), 1).view(x.size(0), -1)

        return box_loc, [
            self.classifier1(rois), self.classifier2(rois), self.classifier3(rois),
            self.classifier4(rois), self.classifier5(rois), self.classifier6(rois),
            self.classifier7(rois),
        ]


@dataclass
class _PassResult:
    indices: list[int]
    confidence: float
    min_char_conf: float
    bbox: list[int]


def _remap_checkpoint(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module.wR2.module."):
            new_key = "wR2." + new_key[len("module.wR2.module.") :]
        elif new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        mapped[new_key] = value
    return mapped


def _to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    tensor = image_bgr.transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(tensor).unsqueeze(0)


def _stretch(image: np.ndarray, size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    return cv2.resize(image, size)


def _letterbox(image: np.ndarray, size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    tw, th = size
    h, w = image.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (nw, nh))
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    y0, x0 = (th - nh) // 2, (tw - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _ccpd_aspect_pad(image: np.ndarray, size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    h, w = image.shape[:2]
    target_h = int(w * CCPD_ASPECT)
    if target_h > h:
        pad = target_h - h
        image = cv2.copyMakeBorder(image, pad // 2, pad - pad // 2, 0, 0, cv2.BORDER_REPLICATE)
    elif target_h < h:
        crop = h - target_h
        image = image[crop // 2 : h - (crop - crop // 2), :]
    return cv2.resize(image, size)


def _enhance(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


def _sharpen(image: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(image, (0, 0), 2.0)
    return cv2.addWeighted(image, 1.5, blur, -0.5, 0)


def _build_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    variants: list[tuple[str, np.ndarray]] = [
        ("stretch", _stretch(image)),
        ("letterbox", _letterbox(image)),
        ("ccpd_aspect", _ccpd_aspect_pad(image)),
        ("clahe", _stretch(_enhance(image))),
        ("sharpen", _stretch(_sharpen(image))),
    ]
    h, w = image.shape[:2]
    if max(h, w) > 640:
        scale = 640 / max(h, w)
        small = cv2.resize(image, (int(w * scale), int(h * scale)))
        variants.append(("downscale", _stretch(small)))
    return variants


def _expand_bbox(bbox: list[int], img_w: int, img_h: int, pad: float = 0.35) -> list[int]:
    x1, y1, x2, y2 = bbox
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    return [
        max(0, int(x1 - bw * pad)),
        max(0, int(y1 - bh * pad)),
        min(img_w, int(x2 + bw * pad)),
        min(img_h, int(y2 + bh * pad)),
    ]


def _bbox_from_box_loc(box: torch.Tensor, img_w: int, img_h: int) -> list[int]:
    cx, cy, bw, bh = box.detach().cpu().tolist()
    x1 = int(max(0, (cx - bw / 2) * img_w))
    y1 = int(max(0, (cy - bh / 2) * img_h))
    x2 = int(min(img_w, (cx + bw / 2) * img_w))
    y2 = int(min(img_h, (cy + bh / 2) * img_h))
    return [x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)]


def _char_vocab_size(pos: int) -> int:
    if pos == 0:
        return PROV_NUM
    if pos == 1:
        return ALPHA_NUM
    return AD_NUM


def _mask_invalid_probs(probs: torch.Tensor, pos: int) -> torch.Tensor:
    masked = probs.clone()
    valid = _char_vocab_size(pos) - 1
    if masked.numel() > valid:
        masked[valid:] = 0.0
    total = masked.sum()
    return masked / total if total > 0 else masked


def _resolve_ambiguous(probs: torch.Tensor, pos: int) -> int:
    if pos < 2:
        return int(probs.argmax().item())
    topv, topi = probs.topk(2)
    i0, i1 = int(topi[0].item()), int(topi[1].item())
    v0, v1 = float(topv[0].item()), float(topv[1].item())
    if v0 > 0 and (v0 - v1) / v0 < 0.25:
        preferred = _DIGIT_PREF.get(frozenset({i0, i1}))
        if preferred is not None:
            return preferred
    return i0


def _decode_pass(char_logits: list[torch.Tensor], src_w: int, src_h: int, box: torch.Tensor) -> _PassResult:
    indices: list[int] = []
    char_confs: list[float] = []
    for pos, logits in enumerate(char_logits):
        probs = _mask_invalid_probs(F.softmax(logits, dim=1)[0].cpu(), pos)
        idx = _resolve_ambiguous(probs, pos)
        indices.append(idx)
        char_confs.append(float(probs[idx].item()))
    return _PassResult(
        indices=indices,
        confidence=float(sum(char_confs) / len(char_confs)),
        min_char_conf=min(char_confs),
        bbox=_bbox_from_box_loc(box, src_w, src_h),
    )


def _pick_bbox(bboxes: list[list[int]], img_w: int, img_h: int) -> list[int]:
    if not bboxes:
        return [0, 0, img_w, img_h]
    scored: list[tuple[float, list[int]]] = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        aspect = bw / max(bh, 1)
        area = bw * bh
        if not (1.8 <= aspect <= 6.5) or area < img_w * img_h * 0.0001:
            continue
        score = area * (1.5 if 2.5 <= aspect <= 5.0 else 1.0)
        scored.append((score, bbox))
    if not scored:
        return bboxes[0]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


class CCPDRPNetRecognizer:
    """CCPD RPNet — runs multiple preprocess variants and keeps the best pass."""

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._model: FH02 | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def model(self) -> FH02:
        if self._model is None:
            net = FH02()
            state = torch.load(self._model_path, map_location="cpu", weights_only=False)
            net.load_state_dict(_remap_checkpoint(state), strict=True)
            net.to(self._device)
            net.eval()
            self._model = net
            logger.info("CCPD RPNet loaded from %s on %s", self._model_path, self._device)
        return self._model

    def _forward(self, image_bgr: np.ndarray) -> tuple[list[torch.Tensor], torch.Tensor, int, int]:
        h, w = image_bgr.shape[:2]
        tensor = _to_tensor(image_bgr).to(self._device)
        with torch.no_grad():
            box_loc, char_logits = self.model(tensor)
        return char_logits, box_loc[0], w, h

    def _evaluate_image(self, image_bgr: np.ndarray, track_bbox: bool) -> tuple[_PassResult | None, list[list[int]]]:
        best: _PassResult | None = None
        bboxes: list[list[int]] = []
        for _, variant in _build_variants(image_bgr):
            logits, box, w, h = self._forward(variant)
            result = _decode_pass(logits, w, h, box)
            if track_bbox:
                bboxes.append(result.bbox)
            text = indices_to_plate(result.indices)
            if len(text) < 7:
                continue
            if best is None or result.confidence > best.confidence:
                best = result
        return best, bboxes

    def recognize(self, image_bgr: np.ndarray) -> dict[str, Any]:
        if image_bgr is None or image_bgr.size == 0:
            return {"plate_number": "", "confidence": 0.0, "bbox": [0, 0, 0, 0], "indices": []}

        img_h, img_w = image_bgr.shape[:2]
        best, all_bboxes = self._evaluate_image(image_bgr, track_bbox=True)

        if best and best.confidence < 0.85:
            seen: set[tuple[int, int, int, int]] = set()
            for bbox in all_bboxes + detect_plate_regions(image_bgr):
                crop_box = _expand_bbox(bbox, img_w, img_h, pad=0.3)
                key = tuple(crop_box)
                if key in seen:
                    continue
                seen.add(key)
                x1, y1, x2, y2 = crop_box
                crop = image_bgr[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 40:
                    continue
                crop_best, _ = self._evaluate_image(crop, track_bbox=False)
                if crop_best and (best is None or crop_best.confidence > best.confidence):
                    best = crop_best

        if best is None:
            return {"plate_number": "", "confidence": 0.0, "bbox": [0, 0, 0, 0], "indices": []}

        plate_number = indices_to_plate(best.indices)
        model_bbox = best.bbox if best.bbox != [0, 0, 1, 1] else _pick_bbox(all_bboxes, img_w, img_h)
        bbox = select_plate_bbox(image_bgr, model_bbox)

        x1, y1, x2, y2 = bbox
        roi = image_bgr[max(0, y1):y2, max(0, x1):x2]
        plate_number = refine_one_vs_i(plate_number, roi)

        return {
            "plate_number": plate_number,
            "confidence": round(best.confidence, 3),
            "bbox": bbox,
            "indices": best.indices,
        }


decode_indices = indices_to_plate

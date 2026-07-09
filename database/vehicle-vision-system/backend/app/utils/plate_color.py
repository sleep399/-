"""车牌颜色识别（基于 ROI 图像分析，不修改 CCPD 文件名车牌内容解析）。"""

import cv2
import numpy as np


PLATE_COLORS = ("蓝牌", "绿牌", "黄牌")


def plate_color_from_path(filename: str) -> str | None:
    """根据 CCPD 子数据集路径推断颜色。"""
    lower = filename.replace("\\", "/").lower()
    if "ccpd_green" in lower or "/green/" in lower:
        return "绿牌"
    if "ccpd_yellow" in lower or "yellow" in lower:
        return "黄牌"
    if "ccpd_blue" in lower or "ccpd_base" in lower or "ccpd_db" in lower:
        return "蓝牌"
    return None


def detect_plate_color(image: np.ndarray, bbox: list[int]) -> str:
    """根据车牌区域 HSV 颜色判断蓝/绿/黄。"""
    if image is None or image.size == 0:
        return "蓝牌"

    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return "蓝牌"

    roi = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    masks = {
        "蓝牌": cv2.inRange(hsv, np.array([95, 50, 40]), np.array([140, 255, 255])),
        "绿牌": cv2.inRange(hsv, np.array([35, 40, 40]), np.array([90, 255, 255])),
        "黄牌": cv2.inRange(hsv, np.array([15, 60, 60]), np.array([40, 255, 255])),
    }
    ratios = {name: float(mask.sum() / 255) / max(mask.size, 1) for name, mask in masks.items()}
    best = max(ratios, key=ratios.get)
    return best if ratios[best] >= 0.12 else "蓝牌"


def resolve_plate_color(image: np.ndarray, bbox: list[int], filename: str = "") -> str:
    """路径提示 + ROI 检测综合判断。"""
    from_path = plate_color_from_path(filename)
    from_roi = detect_plate_color(image, bbox)
    if from_path and from_path == from_roi:
        return from_path
    if from_path in ("绿牌", "黄牌"):
        return from_path
    return from_roi

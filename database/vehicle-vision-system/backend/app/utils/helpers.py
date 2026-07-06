import base64
import re
from typing import Any

# CCPD 官方字符集（与 CCPD-master/rpnet/rpnetEval.py 一致）
CCPD_PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
    "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O",
]
CCPD_ALPHABETS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W",
    "X", "Y", "Z", "O",
]
CCPD_ADS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
    "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "O",
]

COMMON_PROVINCES = "".join(p for p in CCPD_PROVINCES if p != "O")


def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def ndarray_to_base64(img) -> str:
    import cv2
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode("utf-8")


def indices_to_plate(indices: list[int]) -> str:
    """将 CCPD 文件名中的 7 个字符索引解码为车牌号。"""
    if len(indices) < 7:
        return ""
    chars: list[str] = []
    if 0 <= indices[0] < len(CCPD_PROVINCES):
        prov = CCPD_PROVINCES[indices[0]]
        if prov != "O":
            chars.append(prov)
    if 0 <= indices[1] < len(CCPD_ALPHABETS):
        alpha = CCPD_ALPHABETS[indices[1]]
        if alpha != "O":
            chars.append(alpha)
    for idx in indices[2:7]:
        if 0 <= idx < len(CCPD_ADS):
            ch = CCPD_ADS[idx]
            if ch != "O":
                chars.append(ch)
    return "".join(chars)


def parse_ccpd_filename(filename: str) -> dict[str, Any] | None:
    """从 CCPD 数据集文件名解析车牌信息。

    文件名格式（7 段，以 '-' 分隔）:
    Area-Tilt-BBox-Vertices-PlateIndices-Brightness-Blurriness.jpg
    """
    name = filename.replace("\\", "/").split("/")[-1]
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    parts = name.split("-")
    if len(parts) < 7:
        return None

    bbox_part = parts[2]
    try:
        left_up, right_down = [
            [int(v) for v in el.split("&")] for el in bbox_part.split("_")
        ]
    except (ValueError, IndexError):
        return None
    if len(left_up) != 2 or len(right_down) != 2:
        return None
    x1, y1 = left_up
    x2, y2 = right_down

    plate_part = parts[4]
    plate_chars = plate_part.split("_")
    if len(plate_chars) < 7:
        return None
    try:
        indices = [int(c) for c in plate_chars[:7]]
    except ValueError:
        return None

    plate = indices_to_plate(indices)
    if not plate:
        return None

    color = "蓝牌"
    lower_name = filename.lower()
    if "green" in lower_name or "ccpd_green" in lower_name or len(plate) == 8:
        color = "绿牌"
    elif "yellow" in lower_name:
        color = "黄牌"

    return {
        "plate_number": plate,
        "plate_color": color,
        "bbox": [x1, y1, x2, y2],
        "indices": indices,
        "source": "ccpd_filename",
    }


PLATE_COLOR_MAP = {
    "blue": "蓝牌",
    "green": "绿牌",
    "yellow": "黄牌",
    "white": "白牌",
    "black": "黑牌",
}


def detect_plate_color(roi) -> str:
    import cv2
    import numpy as np

    if roi is None or roi.size == 0:
        return "未知"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, w = roi.shape[:2]
    center = hsv[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    mean_h = float(np.mean(center[:, :, 0]))
    mean_s = float(np.mean(center[:, :, 1]))
    mean_v = float(np.mean(center[:, :, 2]))
    if mean_v < 60:
        return "黑牌"
    if mean_s < 40 and mean_v > 180:
        return "白牌"
    # 蓝牌优先于绿牌（避免蓝色区域被误判为绿牌）
    if 95 <= mean_h <= 140 and mean_s > 45:
        return "蓝牌"
    if 35 <= mean_h <= 85 and mean_s > 50 and mean_v > 80:
        return "绿牌"
    if 20 <= mean_h <= 35 and mean_s > 80:
        return "黄牌"
    return "蓝牌"


def _blue_pixel_ratio(image, bbox: list[int]) -> float:
    """计算框内蓝色像素占比，用于验证是否为真实蓝牌区域。"""
    import cv2
    import numpy as np

    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([95, 50, 40]), np.array([140, 255, 255]))
    return float(mask.sum() / 255) / max(mask.size, 1)


def _find_blue_plate_contours(image) -> list[list[int]]:
    """返回蓝色车牌紧凑外接矩形（不加过多 padding）。"""
    import cv2
    import numpy as np

    if image is None or image.size == 0:
        return []
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 两段蓝色范围，兼容偏浅/偏深的蓝牌
    mask = cv2.inRange(hsv, np.array([95, 50, 40]), np.array([140, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([100, 80, 60]), np.array([130, 255, 220]))
    mask = cv2.bitwise_or(mask, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[list[int]] = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / max(ch, 1)
        area = cw * ch
        if 2.0 <= aspect <= 6.5 and area >= w * h * 0.00015 and ch >= 6:
            boxes.append([x, y, x + cw, y + ch])
    return boxes


def _scan_lower_half_for_plate(image) -> list[int] | None:
    """在图像下半部扫描蓝牌，用于模型框或上半部误检时的兜底。"""
    h, w = image.shape[:2]
    y0 = int(h * 0.38)
    lower = image[y0:, :]
    best: tuple[float, list[int]] | None = None
    for x1, y1, x2, y2 in _find_blue_plate_contours(lower):
        bbox = [x1, y1 + y0, x2, y2 + y0]
        ratio = _blue_pixel_ratio(image, bbox)
        bw, bh = x2 - x1, y2 - y1
        aspect = bw / max(bh, 1)
        cy = (bbox[1] + bbox[3]) / 2 / h
        if ratio < 0.30 or not (2.2 <= aspect <= 6.0):
            continue
        score = ratio * 1000 + cy * 200 + min(bw * bh, 8000) * 0.05
        if best is None or score > best[0]:
            best = (score, bbox)
    return best[1] if best else None


def _pad_bbox(bbox: list[int], img_w: int, img_h: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    pad_x, pad_y = int((x2 - x1) * 0.04), int((y2 - y1) * 0.12)
    return [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(img_w, x2 + pad_x),
        min(img_h, y2 + pad_y),
    ]


def select_plate_bbox(image, fallback_bbox: list[int] | None = None) -> list[int]:
    """综合蓝色区域检测与模型框，选取最可能的车牌位置。"""
    if image is None or image.size == 0:
        return fallback_bbox or [0, 0, 0, 0]

    img_h, img_w = image.shape[:2]
    blue_boxes = _find_blue_plate_contours(image)
    candidates: list[list[int]] = list(blue_boxes)

    if fallback_bbox and len(fallback_bbox) == 4:
        fx1, fy1, fx2, fy2 = fallback_bbox
        if fx2 > fx1 and fy2 > fy1:
            candidates.append([fx1, fy1, fx2, fy2])

    best_score = -1.0
    best_bbox: list[int] | None = None

    for x1, y1, x2, y2 in candidates:
        bw, bh = x2 - x1, y2 - y1
        bbox = [x1, y1, x2, y2]
        aspect = bw / max(bh, 1)
        cy = (y1 + y2) / 2 / img_h
        blue_ratio = _blue_pixel_ratio(image, bbox)

        if not (1.8 <= aspect <= 6.5):
            continue
        if blue_ratio < 0.22 and cy < 0.45:
            continue

        score = blue_ratio * 800.0 + min(bw * bh, 12000) * 0.04
        if cy >= 0.50:
            score *= 4.0
        elif cy >= 0.42:
            score *= 2.0
        else:
            score *= 0.04
        if 2.5 <= aspect <= 5.5:
            score *= 1.4
        if blue_ratio >= 0.45:
            score *= 1.5

        if score > best_score:
            best_score = score
            best_bbox = bbox

    if best_bbox is None:
        best_bbox = _scan_lower_half_for_plate(image)

    if best_bbox is None and fallback_bbox and len(fallback_bbox) == 4:
        fx1, fy1, fx2, fy2 = fallback_bbox
        if fx2 > fx1 and fy2 > fy1:
            best_bbox = [fx1, fy1, fx2, fy2]

    if best_bbox is None:
        return fallback_bbox or [0, 0, img_w, img_h]

    padded = _pad_bbox(best_bbox, img_w, img_h)
    roi = image[padded[1]:padded[3], padded[0]:padded[2]]
    if roi.size > 0 and detect_plate_color(roi) == "黑牌":
        lower_bbox = _scan_lower_half_for_plate(image)
        if lower_bbox is not None:
            padded = _pad_bbox(lower_bbox, img_w, img_h)

    return padded


def _char_cell_slices(width: int) -> list[tuple[int, int]]:
    """蓝牌 7 位字符的大致水平分区（省份字略宽）。"""
    ratios = [0.0, 0.14, 0.26, 0.38, 0.50, 0.62, 0.74, 0.86, 1.0]
    return [(int(width * ratios[i]), int(width * ratios[i + 1])) for i in range(7)]


def _extract_char_ink(cell) -> "np.ndarray":
    """提取车牌字符笔画（蓝底白字）。"""
    import cv2
    import numpy as np

    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if cell.ndim == 3 else cell
    gray = cv2.resize(gray, (32, 64))
    # 蓝牌为白色字符，取高亮区域
    _, bright = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    ink = bright > 0
    if ink.sum() < 15:
        _, ink_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ink = ink_mask > 0 if ink_mask.mean() > 127 else ink_mask < 128
    return ink


def _looks_like_letter_i(cell) -> bool:
    """判断单个字符区域更像字母 I 还是数字 1。"""
    import numpy as np

    if cell is None or cell.size == 0:
        return False
    ink = _extract_char_ink(cell)
    if ink.sum() < 20:
        return False

    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return False

    ink_h = rows[-1] - rows[0] + 1
    ink_w = cols[-1] - cols[0] + 1
    ratio = ink_w / max(ink_h, 1)

    h = ink.shape[0]
    top = float(ink[: h // 5, :].sum())
    mid = float(ink[2 * h // 5 : 3 * h // 5, :].sum())
    bot = float(ink[4 * h // 5 :, :].sum())
    total = top + mid + bot + 1.0

    # 数字 1：中间细、上下有较粗笔画（衬线/底座）
    has_serif = (top / total > 0.22 and bot / total > 0.22) or bot > mid * 1.8
    if has_serif:
        return False
    # 字母 I：竖线均匀，中间与上下宽度接近
    if ratio < 0.42 and mid / total > 0.18:
        return True
    return ratio < 0.30


def refine_one_vs_i(plate_text: str, roi) -> str:
    """对识别结果中的数字 1 做字形复核。

    中国蓝牌不使用字母 I/O，模型只能输出数字 1。
    若字符为竖线且无上下衬线，则显示为 I 以便与 1 区分。
    """
    if len(plate_text) != 7 or roi is None or roi.size == 0:
        return plate_text

    h, w = roi.shape[:2]
    if h < 8 or w < 40:
        return plate_text

    margin = max(2, int(w * 0.02))
    inner = roi[:, margin : w - margin]
    slices = _char_cell_slices(inner.shape[1])

    chars = list(plate_text)
    for i in range(2, 7):
        if chars[i] != "1":
            continue
        x1, x2 = slices[i]
        if x2 <= x1:
            continue
        cell = inner[:, x1:x2]
        if _looks_like_letter_i(cell):
            chars[i] = "I"
    return "".join(chars)


def detect_plate_regions(image) -> list[list[int]]:
    """基于蓝色车牌颜色的候选区域检测（带 padding，用于裁剪精识别）。"""
    import cv2

    if image is None or image.size == 0:
        return []
    h, w = image.shape[:2]
    regions: list[tuple[float, list[int]]] = []
    for x1, y1, x2, y2 in _find_blue_plate_contours(image):
        cw, ch = x2 - x1, y2 - y1
        pad_x, pad_y = 0.15, 0.35
        bbox = [
            max(0, int(x1 - cw * pad_x)),
            max(0, int(y1 - ch * pad_y)),
            min(w, int(x2 + cw * pad_x)),
            min(h, int(y2 + ch * pad_y)),
        ]
        regions.append((cw * ch, bbox))
    regions.sort(key=lambda item: item[0], reverse=True)
    return [bbox for _, bbox in regions[:3]]


def clean_plate_text(text: str) -> str:
    """保留车牌合法字符（中文省份 + 字母 + 数字）。"""
    text = re.sub(r"[^0-9A-Z\u4e00-\u9fff]", "", text.upper())
    if text and text[0] not in COMMON_PROVINCES:
        for p in COMMON_PROVINCES:
            if p in text:
                text = p + text.replace(p, "", 1)
                break
    return text[:8]


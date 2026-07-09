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
    elif "white" in lower_name:
        color = "白牌"
    elif "black" in lower_name:
        color = "黑牌"

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

# 各色车牌 HSV 掩码与文字极性（light=浅色字 on 深色底，dark=深色字 on 浅色底）
PLATE_COLOR_SPECS: dict[str, dict[str, Any]] = {
    "蓝牌": {
        "masks": [
            ([95, 50, 40], [140, 255, 255]),
            ([100, 80, 60], [130, 255, 220]),
        ],
        "min_ratio": 0.22,
        "text_mode": "light",
    },
    "绿牌": {
        "masks": [
            ([35, 40, 50], [90, 255, 255]),
            ([40, 30, 80], [85, 200, 220]),
        ],
        "min_ratio": 0.20,
        "text_mode": "light",
    },
    "黄牌": {
        "masks": [
            ([15, 80, 80], [40, 255, 255]),
            ([20, 60, 100], [35, 255, 255]),
        ],
        "min_ratio": 0.25,
        "text_mode": "dark",
    },
    "白牌": {
        "masks": [
            ([0, 0, 175], [180, 55, 255]),
        ],
        "min_ratio": 0.32,
        "text_mode": "dark",
    },
    "黑牌": {
        "masks": [
            ([0, 0, 0], [180, 255, 75]),
        ],
        "min_ratio": 0.28,
        "text_mode": "light",
    },
}


def _build_plate_color_mask(hsv, plate_color: str):
    """根据车牌颜色规格生成二值掩码。"""
    import cv2
    import numpy as np

    spec = PLATE_COLOR_SPECS.get(plate_color)
    if not spec:
        return np.zeros(hsv.shape[:2], dtype=np.uint8)
    mask = None
    for low, high in spec["masks"]:
        part = cv2.inRange(hsv, np.array(low), np.array(high))
        mask = part if mask is None else cv2.bitwise_or(mask, part)
    return mask if mask is not None else np.zeros(hsv.shape[:2], dtype=np.uint8)


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
    if mean_s < 50 and mean_v > 165:
        return "白牌"
    # 蓝牌优先于绿牌（避免蓝色区域被误判为绿牌）
    if 95 <= mean_h <= 140 and mean_s > 45:
        return "蓝牌"
    if 35 <= mean_h <= 85 and mean_s > 50 and mean_v > 80:
        return "绿牌"
    if 12 <= mean_h <= 40 and mean_s > 70:
        return "黄牌"
    if mean_v < 95 and mean_s < 85:
        return "黑牌"
    dominant = _dominant_plate_color(roi, [0, 0, w, h])
    return dominant if dominant != "未知" else "蓝牌"


def _color_pixel_ratio(image, bbox: list[int], plate_color: str) -> float:
    """计算框内指定颜色车牌像素占比。"""
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
    mask = _build_plate_color_mask(hsv, plate_color)
    return float(mask.sum() / 255) / max(mask.size, 1)


def _blue_pixel_ratio(image, bbox: list[int]) -> float:
    """计算框内蓝色像素占比（兼容旧接口）。"""
    return _color_pixel_ratio(image, bbox, "蓝牌")


def _dominant_plate_color(image, bbox: list[int]) -> str:
    """根据像素占比推断框内最可能的车牌颜色。"""
    best_color = "蓝牌"
    best_ratio = 0.0
    for color in PLATE_COLOR_SPECS:
        ratio = _color_pixel_ratio(image, bbox, color)
        if ratio > best_ratio:
            best_ratio = ratio
            best_color = color
    return best_color if best_ratio >= 0.15 else "未知"


def _text_on_plate_ratio(image, bbox: list[int], plate_color: str) -> float:
    """车牌文字对比度：浅色字 on 深色底，或深色字 on 浅色底。"""
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
    plate_mask = _build_plate_color_mask(hsv, plate_color)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    spec = PLATE_COLOR_SPECS.get(plate_color, PLATE_COLOR_SPECS["蓝牌"])
    plate_count = max(int(plate_mask.sum() / 255), 1)
    if spec["text_mode"] == "dark":
        text = gray < 115
    else:
        text = gray > 145
    return float(np.logical_and(plate_mask > 0, text).sum()) / plate_count


def _white_text_on_blue_ratio(image, bbox: list[int]) -> float:
    """蓝底白字特征（兼容旧接口）。"""
    return _text_on_plate_ratio(image, bbox, "蓝牌")


def _plate_aspect_ok(bw: int, bh: int, plate_color: str) -> bool:
    aspect = bw / max(bh, 1)
    if plate_color == "绿牌":
        return 1.5 <= aspect <= 7.5
    return 1.7 <= aspect <= 6.5


def _find_colored_plate_contours(image, plate_color: str) -> list[list[int]]:
    """按指定颜色查找车牌外接矩形。"""
    import cv2

    if image is None or image.size == 0:
        return []
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = _build_plate_color_mask(hsv, plate_color)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[list[int]] = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area > w * h * 0.25:
            continue
        if _plate_aspect_ok(cw, ch, plate_color) and area >= w * h * 0.00015 and ch >= 6:
            boxes.append([x, y, x + cw, y + ch])
    return boxes


def _find_blue_plate_contours(image) -> list[list[int]]:
    """返回蓝色车牌紧凑外接矩形（兼容旧接口）。"""
    return _find_colored_plate_contours(image, "蓝牌")


def _find_all_plate_contours(image) -> list[tuple[str, list[int]]]:
    """返回各颜色候选车牌框。"""
    results: list[tuple[str, list[int]]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for color in PLATE_COLOR_SPECS:
        for bbox in _find_colored_plate_contours(image, color):
            key = tuple(bbox)
            if key in seen:
                continue
            seen.add(key)
            results.append((color, bbox))
    return results


def _score_plate_bbox(image, bbox: list[int], plate_color: str | None = None) -> float:
    """根据车牌视觉特征打分，支持蓝/绿/黄/白/黑牌。"""
    x1, y1, x2, y2 = bbox
    img_h, img_w = image.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    if bw < 60 or bh < 12:
        return -1.0

    if plate_color is None or plate_color not in PLATE_COLOR_SPECS:
        plate_color = _dominant_plate_color(image, bbox)
    if plate_color == "未知":
        return -1.0

    if not _plate_aspect_ok(bw, bh, plate_color):
        return -1.0

    area_ratio = (bw * bh) / max(img_w * img_h, 1)
    if area_ratio > 0.12:
        return -1.0

    spec = PLATE_COLOR_SPECS[plate_color]
    color_ratio = _color_pixel_ratio(image, bbox, plate_color)
    text_ratio = _text_on_plate_ratio(image, bbox, plate_color)

    if color_ratio < spec["min_ratio"]:
        return -1.0
    min_text = 0.015 if color_ratio >= 0.38 else 0.04
    if text_ratio < min_text and color_ratio < 0.38:
        return -1.0
    if area_ratio < 0.008 and (bw < 100 or text_ratio > 0.85):
        return -1.0

    score = color_ratio * 350.0 + text_ratio * 450.0 + color_ratio * text_ratio * 500.0
    if 2.3 <= (bw / max(bh, 1)) <= 5.5:
        score += 50.0
    if color_ratio >= 0.40:
        score += 60.0
    if text_ratio >= 0.08:
        score += 80.0
    if 0.002 <= area_ratio <= 0.06:
        score += 30.0
    if area_ratio > 0.08:
        score -= 60.0
    return score


def _pad_bbox(bbox: list[int], img_w: int, img_h: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    pad_x, pad_y = int((x2 - x1) * 0.04), int((y2 - y1) * 0.12)
    return [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(img_w, x2 + pad_x),
        min(img_h, y2 + pad_y),
    ]


def bbox_from_rpnet_box(box, orig_w: int, orig_h: int) -> list[int]:
    """CCPD demo.py：归一化 (cx,cy,w,h) 映射回原图坐标。"""
    if hasattr(box, "detach"):
        cx, cy, bw, bh = box.detach().cpu().tolist()
    else:
        cx, cy, bw, bh = box
    x1 = int(max(0, (cx - bw / 2) * orig_w))
    y1 = int(max(0, (cy - bh / 2) * orig_h))
    x2 = int(min(orig_w, (cx + bw / 2) * orig_w))
    y2 = int(min(orig_h, (cy + bh / 2) * orig_h))
    return [x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)]


def _is_valid_plate_roi(image, bbox: list[int]) -> bool:
    x1, y1, x2, y2 = bbox
    roi = image[max(0, y1):y2, max(0, x1):x2]
    if roi.size == 0:
        return False
    color = detect_plate_color(roi)
    if color in PLATE_COLOR_SPECS:
        return _color_pixel_ratio(image, bbox, color) >= PLATE_COLOR_SPECS[color]["min_ratio"] * 0.85
    for plate_color, spec in PLATE_COLOR_SPECS.items():
        if _color_pixel_ratio(image, bbox, plate_color) >= spec["min_ratio"]:
            return True
    return False


def _best_plate_bbox(image) -> list[int] | None:
    """在所有颜色候选中选取得分最高的车牌框。"""
    best_score = -1.0
    best_bbox: list[int] | None = None
    for color, bbox in _find_all_plate_contours(image):
        score = _score_plate_bbox(image, bbox, plate_color=color)
        if score > best_score:
            best_score = score
            best_bbox = bbox
    return best_bbox


def _best_blue_plate_bbox(image) -> list[int] | None:
    """兼容旧接口。"""
    return _best_plate_bbox(image)


def locate_plate_bbox(
    image,
    rpnet_bbox: list[int],
    ccpd_bbox: list[int] | None = None,
) -> list[int]:
    """定位车牌框：CCPD 标注 > RPNet 定位 > 多颜色区域兜底。"""
    if image is None or image.size == 0:
        return rpnet_bbox or [0, 0, 0, 0]

    img_h, img_w = image.shape[:2]

    if ccpd_bbox and len(ccpd_bbox) == 4:
        x1, y1, x2, y2 = ccpd_bbox
        if x2 > x1 and y2 > y1:
            return _pad_bbox([x1, y1, x2, y2], img_w, img_h)

    if _is_valid_plate_roi(image, rpnet_bbox):
        return _pad_bbox(rpnet_bbox, img_w, img_h)

    color_bbox = _best_plate_bbox(image)
    if color_bbox is not None:
        return _pad_bbox(color_bbox, img_w, img_h)

    return _pad_bbox(rpnet_bbox, img_w, img_h)


def select_plate_bbox(
    image,
    fallback_bbox: list[int] | None = None,
    extra_bboxes: list[list[int]] | None = None,
) -> list[int]:
    """兼容旧接口，内部转 locate_plate_bbox。"""
    return locate_plate_bbox(image, fallback_bbox or [0, 0, 0, 0])


def _char_cell_slices(width: int, plate_color: str = "蓝牌") -> list[tuple[int, int]]:
    """车牌字符水平分区（省份字略宽；绿牌新能源可能 8 位）。"""
    if plate_color == "绿牌" and width > 0:
        ratios = [0.0, 0.11, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0]
        count = 8
    else:
        ratios = [0.0, 0.14, 0.26, 0.38, 0.50, 0.62, 0.74, 0.86, 1.0]
        count = 7
    return [(int(width * ratios[i]), int(width * ratios[i + 1])) for i in range(count)]


def _is_dark_text_plate(plate_color: str) -> bool:
    spec = PLATE_COLOR_SPECS.get(plate_color, PLATE_COLOR_SPECS["蓝牌"])
    return spec["text_mode"] == "dark"


def _extract_char_ink(cell, plate_color: str = "蓝牌") -> "np.ndarray":
    """提取车牌字符笔画（自动适配深浅底色）。"""
    import cv2
    import numpy as np

    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if cell.ndim == 3 else cell
    gray = cv2.resize(gray, (32, 64))
    if _is_dark_text_plate(plate_color):
        _, ink_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink = ink_mask > 0 if ink_mask.mean() > 127 else ink_mask < 128
        if ink.sum() < 15:
            ink = gray < 120
    else:
        _, bright = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        ink = bright > 0
        if ink.sum() < 15:
            _, ink_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ink = ink_mask > 0 if ink_mask.mean() > 127 else ink_mask < 128
    return ink


def _trim_char_cell(cell):
    """裁掉字符格左右边缘，减少相邻字符干扰。"""
    if cell is None or cell.size == 0:
        return cell
    cw = cell.shape[1]
    if cw < 6:
        return cell
    margin = max(1, int(cw * 0.12))
    return cell[:, margin : cw - margin]


def _stroke_band_widths(ink) -> tuple[int, int, int]:
    """上/中/下三段笔画水平宽度。"""
    import numpy as np

    h = ink.shape[0]

    def band(y0: int, y1: int) -> int:
        cols = np.where(ink[y0:y1].any(axis=0))[0]
        return int(len(cols)) if len(cols) else 0

    return band(0, h // 3), band(h // 3, 2 * h // 3), band(2 * h // 3, h)


def _center_column_fill_ratio(ink) -> float:
    """中心竖线连续填充比例，字母 I 通常较高。"""
    import numpy as np

    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0.0
    cx = (cols[0] + cols[-1]) // 2
    w = ink.shape[1]
    band = ink[:, max(0, cx - 1) : min(w, cx + 2)]
    active = band.any(axis=1)
    active_rows = np.where(active)[0]
    if len(active_rows) == 0:
        return 0.0
    return (active_rows[-1] - active_rows[0] + 1) / max(ink.shape[0], 1)


def _looks_like_digit_one(ink) -> bool:
    """数字 1 常见上下衬线/底座的笔画特征。"""
    tw, mw, bw = _stroke_band_widths(ink)
    if mw == 0:
        return bw > tw and bw >= 3
    if bw > mw * 1.25 and tw < mw * 0.7:
        return True
    if tw > mw * 1.25 and bw < mw * 0.7:
        return True
    return False


def _looks_like_letter_i(cell, plate_color: str = "蓝牌") -> bool:
    """判断单个字符区域更像字母 I 还是数字 1。"""
    import numpy as np

    if cell is None or cell.size == 0:
        return False
    cell = _trim_char_cell(cell)
    ink = _extract_char_ink(cell, plate_color)
    if ink.sum() < 20:
        return False

    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return False

    ink_h = rows[-1] - rows[0] + 1
    ink_w = cols[-1] - cols[0] + 1
    ratio = ink_w / max(ink_h, 1)
    fill = _center_column_fill_ratio(ink)
    tw, mw, bw = _stroke_band_widths(ink)

    if _looks_like_digit_one(ink):
        return False

    # 字母 I：竖线连续、上中下宽度均匀；数字 1 通常带底衬或上钩
    if fill >= 0.62 and ratio <= 0.68 and mw >= 3:
        widths = [tw, mw, bw]
        if max(widths) <= min(widths) * 1.35 + 1:
            return True
    return False


def refine_one_vs_i(plate_text: str, roi, plate_color: str = "蓝牌") -> str:
    """对识别结果中的数字 1 做字形复核。

    模型字符集不含字母 I，只能输出数字 1。
    若字符为竖线且无上下衬线，则显示为 I 以便与 1 区分。
    """
    if len(plate_text) not in (7, 8) or roi is None or roi.size == 0:
        return plate_text

    if plate_color == "未知":
        plate_color = detect_plate_color(roi)

    h, w = roi.shape[:2]
    if h < 8 or w < 40:
        return plate_text

    margin = max(2, int(w * 0.02))
    inner = roi[:, margin : w - margin]
    slices = _char_cell_slices(inner.shape[1], plate_color)

    chars = list(plate_text)
    start = 2 if len(plate_text) == 7 else 2
    end = min(len(chars), len(slices))
    for i in range(start, end):
        if chars[i] != "1":
            continue
        x1, x2 = slices[i]
        if x2 <= x1:
            continue
        cell = inner[:, x1:x2]
        if _looks_like_letter_i(cell, plate_color):
            chars[i] = "I"
    return "".join(chars)


def detect_plate_regions(image) -> list[list[int]]:
    """基于多颜色车牌的候选区域检测（带 padding，用于裁剪精识别）。"""
    if image is None or image.size == 0:
        return []
    h, w = image.shape[:2]
    regions: list[tuple[float, list[int]]] = []
    for _, (x1, y1, x2, y2) in _find_all_plate_contours(image):
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
    return [bbox for _, bbox in regions[:5]]


def clean_plate_text(text: str) -> str:
    """保留车牌合法字符（中文省份 + 字母 + 数字）。"""
    text = re.sub(r"[^0-9A-Z\u4e00-\u9fff]", "", text.upper())
    if text and text[0] not in COMMON_PROVINCES:
        for p in COMMON_PROVINCES:
            if p in text:
                text = p + text.replace(p, "", 1)
                break
    return text[:8]


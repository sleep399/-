"""CCPD-master/rpnet/load_data.py 标注解析 + 预处理。"""

from __future__ import annotations

import cv2
import numpy as np

from app.ccpd.charset import ads, alphabets, provinces

imgSize = (480, 480)


def stem_from_img_path(img_name: str) -> str:
    return img_name.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]


def iname_from_img_path(img_name: str) -> list[str]:
    return stem_from_img_path(img_name).split("-")


def label_from_img_path(img_name: str) -> str:
    return stem_from_img_path(img_name).split("-")[-3]


def label_indices_from_img_path(img_name: str) -> list[int]:
    """rpnetEval.py: 比较用前 7 位。"""
    return [int(ee) for ee in label_from_img_path(img_name).split("_")[:7]]


def label_indices_all_from_img_path(img_name: str) -> list[int]:
    """文件名中全部车牌字符索引（绿牌 8 位）。"""
    return [int(ee) for ee in label_from_img_path(img_name).split("_")]


def indices_to_plate_from_list(indices: list[int]) -> str:
    if len(indices) < 7:
        return ""
    plate = (
        provinces[indices[0]]
        + alphabets[indices[1]]
        + ads[indices[2]]
        + ads[indices[3]]
        + ads[indices[4]]
        + ads[indices[5]]
        + ads[indices[6]]
    )
    for idx in indices[7:]:
        plate += ads[idx]
    return plate


def bbox_from_iname(iname: list[str]) -> tuple[list[int], list[int]]:
    left_up, right_down = [
        [int(eel) for eel in el.split("&")] for el in iname[2].split("_")
    ]
    return left_up, right_down


def vertices_from_iname(iname: list[str]) -> list[list[int]]:
    return [[int(eel) for eel in el.split("&")] for el in iname[3].split("_")]


def preprocess_image_bgr(img: np.ndarray, img_size: tuple[int, int] = imgSize) -> np.ndarray:
    resized = cv2.resize(img, img_size)
    resized = np.transpose(resized, (2, 0, 1))
    resized = resized.astype("float32")
    resized /= 255.0
    return resized


def parse_ccpd_filename(filename: str) -> dict | None:
    """README: 文件名恰好 7 段。"""
    parts = stem_from_img_path(filename).split("-")
    if len(parts) != 7:
        return None

    area, tilt, _box, _verts, plate_label, brightness_str, blurriness_str = parts
    iname = parts

    try:
        left_up, right_down = bbox_from_iname(iname)
        vertices = vertices_from_iname(iname)
        indices_all = label_indices_all_from_img_path(filename)
        indices_7 = indices_all[:7]
        horizontal_tilt, vertical_tilt = tilt.split("_")
        brightness = int(brightness_str)
        blurriness = int(blurriness_str)
    except (ValueError, IndexError):
        return None

    if len(left_up) != 2 or len(right_down) != 2 or len(indices_7) != 7:
        return None

    return {
        "area": area,
        "tilt": tilt,
        "horizontal_tilt": horizontal_tilt,
        "vertical_tilt": vertical_tilt,
        "bbox": [left_up[0], left_up[1], right_down[0], right_down[1]],
        "vertices": vertices,
        "plate_label": plate_label,
        "indices": indices_all,
        "indices_7": indices_7,
        "plate_number": indices_to_plate_from_list(indices_all),
        "brightness": brightness,
        "blurriness": blurriness,
        "source": "ccpd_filename",
    }


ccpd_stem = stem_from_img_path
ccpd_iname_parts = iname_from_img_path
ccpd_plate_label_str = label_from_img_path
ccpd_plate_indices = label_indices_from_img_path

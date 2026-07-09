"""在 OpenCV 图像上绘制中文文字（cv2.putText 不支持中文会显示为 ?）。"""

from __future__ import annotations

import os
import platform
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=1)
def _cn_font_path() -> str | None:
    candidates: list[Path] = []
    if platform.system() == "Windows":
        windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
        candidates.extend([
            windir / "Fonts" / "msyh.ttc",
            windir / "Fonts" / "msyhbd.ttc",
            windir / "Fonts" / "simhei.ttf",
            windir / "Fonts" / "simsun.ttc",
        ])
    candidates.extend([
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    ])
    for path in candidates:
        if path.exists():
            return str(path)
    return None


@lru_cache(maxsize=8)
def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = _cn_font_path()
    if path:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_cn_text_bgr(
    image_bgr: np.ndarray,
    text: str,
    org: tuple[int, int],
    color_bgr: tuple[int, int, int],
    font_size: int = 22,
) -> np.ndarray:
    """在 BGR 图像上绘制中文标签，返回 BGR 图像。"""
    if not text:
        return image_bgr

    x, y = org
    rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    font = _load_font(font_size)

    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        text_w, text_h = draw.textsize(text, font=font)

    pad = 4
    top = max(y - text_h - pad * 2, 0)
    left = max(x, 0)
    draw.rectangle(
        [left, top, left + text_w + pad * 2, top + text_h + pad * 2],
        fill=(0, 0, 0),
    )
    draw.text((left + pad, top + pad), text, font=font, fill=rgb)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

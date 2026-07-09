from __future__ import annotations
import base64
import cv2
import numpy as np


def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def ndarray_to_base64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise ValueError("Unable to encode image")
    return base64.b64encode(buf.tobytes()).decode("utf-8")

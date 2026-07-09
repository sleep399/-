import base64

from app.ccpd.decode import indices_to_plate
from app.ccpd.load_data import parse_ccpd_filename

__all__ = [
    "image_to_base64",
    "ndarray_to_base64",
    "indices_to_plate",
    "parse_ccpd_filename",
]


def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def ndarray_to_base64(img) -> str:
    import cv2
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode("utf-8")

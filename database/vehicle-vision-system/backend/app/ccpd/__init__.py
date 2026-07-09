"""CCPD-master/rpnet 官方逻辑移植。"""

from app.ccpd.charset import ads, alphabets, provinces
from app.ccpd.decode import indices_to_plate, is_equal, label_pred_from_logits
from app.ccpd.inference import CCPDRPNetRecognizer
from app.ccpd.load_data import label_indices_from_img_path, parse_ccpd_filename, preprocess_image_bgr

__all__ = [
    "provinces",
    "alphabets",
    "ads",
    "CCPDRPNetRecognizer",
    "indices_to_plate",
    "is_equal",
    "label_pred_from_logits",
    "parse_ccpd_filename",
    "label_indices_from_img_path",
    "preprocess_image_bgr",
]

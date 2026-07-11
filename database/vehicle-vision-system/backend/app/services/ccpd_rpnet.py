"""CCPD RPNet — 转发至 app.ccpd（与 CCPD-master 一致）。"""

from app.ccpd import CCPDRPNetRecognizer, indices_to_plate

decode_indices = indices_to_plate

__all__ = ["CCPDRPNetRecognizer", "indices_to_plate", "decode_indices"]

from pathlib import Path
from typing import Any
import logging

import cv2
import numpy as np

from app.config import settings
from app.services.ccpd_rpnet import CCPDRPNetRecognizer
from app.utils.helpers import (
    detect_plate_color,
    ndarray_to_base64,
    parse_ccpd_filename,
    select_plate_bbox,
)
from app.utils.model_loader import get_model_path

logger = logging.getLogger(__name__)


class LicensePlateService:
    """车牌检测与识别服务（CCPD RPNet 端到端模型）。"""

    def __init__(self):
        self._rpnet: CCPDRPNetRecognizer | None = None

    @property
    def rpnet(self) -> CCPDRPNetRecognizer:
        if self._rpnet is None:
            model_path = get_model_path("fh02.pth")
            self._rpnet = CCPDRPNetRecognizer(model_path)
        return self._rpnet

    def _recognize_with_rpnet(self, image: np.ndarray) -> dict[str, Any]:
        result = self.rpnet.recognize(image)
        plate_number = result.get("plate_number", "")
        if len(plate_number) < 7:
            return {"plate_number": "", "confidence": 0.0, "bbox": result.get("bbox", [0, 0, 0, 0])}

        bbox = select_plate_bbox(image, result["bbox"])
        x1, y1, x2, y2 = bbox
        roi = image[max(0, y1):y2, max(0, x1):x2]
        return {
            "plate_number": plate_number,
            "plate_color": detect_plate_color(roi) if roi.size > 0 else "蓝牌",
            "confidence": result["confidence"],
            "bbox": bbox,
        }

    def recognize(self, image_bytes: bytes, filename: str = "") -> dict[str, Any]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法解析图像")

        plates: list[dict] = []
        ccpd_info = parse_ccpd_filename(filename) if filename else None

        try:
            rpnet_result = self._recognize_with_rpnet(image)
        except Exception as exc:
            logger.exception("RPNet 识别失败: %s", exc)
            rpnet_result = {"plate_number": "", "confidence": 0.0, "bbox": [0, 0, 0, 0]}

        if ccpd_info:
            plate_num = rpnet_result["plate_number"]
            conf = rpnet_result["confidence"]
            bbox = rpnet_result["bbox"] if rpnet_result["plate_number"] else ccpd_info["bbox"]
            if len(plate_num) < 7:
                plate_num = ccpd_info["plate_number"]
                conf = max(conf, 0.9)
                bbox = ccpd_info["bbox"]
            plates.append({
                "plate_number": plate_num,
                "plate_color": ccpd_info["plate_color"],
                "confidence": round(conf, 3),
                "bbox": bbox,
            })
        elif rpnet_result["plate_number"] and rpnet_result["confidence"] >= settings.lpr_min_confidence:
            plates.append(rpnet_result)

        annotated = image.copy()
        for p in plates:
            x1, y1, x2, y2 = p["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{p['plate_number']} ({p['plate_color']})"
            cv2.putText(
                annotated, label, (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

        return {
            "plates": plates,
            "plate_count": len(plates),
            "annotated_image": ndarray_to_base64(annotated),
            "success": len(plates) > 0,
        }

    def recognize_frame(self, frame: np.ndarray, frame_index: int = 0) -> dict[str, Any]:
        _, buf = cv2.imencode(".jpg", frame)
        return self.recognize(buf.tobytes(), filename=f"frame_{frame_index}.jpg")

    def process_video(self, video_path: Path, sample_interval: int = 15) -> list[dict]:
        cap = cv2.VideoCapture(str(video_path))
        results = []
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_interval == 0:
                res = self.recognize_frame(frame, idx)
                if res["plates"]:
                    results.append({"frame": idx, **res})
            idx += 1
        cap.release()
        return results


lpr_service = LicensePlateService()

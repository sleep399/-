from pathlib import Path
from typing import Any
import logging

import numpy as np

from app.ccpd.load_data import parse_ccpd_filename
from app.utils.helpers import ndarray_to_base64
from app.utils.image_draw import draw_cn_text_bgr
from app.utils.plate_color import resolve_plate_color

try:
    import cv2
except Exception as exc:  # pragma: no cover - environment dependent
    cv2 = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

ANNOTATE_COLORS = {
    "蓝牌": (255, 128, 0),
    "绿牌": (0, 220, 0),
    "黄牌": (0, 220, 255),
}


class LicensePlateService:
    """车牌识别：CCPD 图片走 GT；视频/普通图片走本地 YOLO+LPRNet 资产。"""

    def __init__(self) -> None:
        self._rpnet = None
        self._rpnet_error: str | None = None
        self._yolo_video = None
        self._yolo_video_error: str | None = None

    def _get_rpnet(self):
        if self._rpnet is None and self._rpnet_error is None:
            try:
                from app.ccpd.inference import CCPDRPNetRecognizer
                from app.utils.model_loader import get_model_path

                self._rpnet = CCPDRPNetRecognizer(get_model_path("fh02.pth"))
            except Exception as exc:
                self._rpnet_error = str(exc)
                logger.warning("RPNet 未加载: %s", exc)
        return self._rpnet

    def _get_yolo_video(self):
        if self._yolo_video is None and self._yolo_video_error is None:
            try:
                from app.yolo_lprnet.pipeline import YoloLprPipeline
                self._yolo_video = YoloLprPipeline()
            except Exception as exc:
                self._yolo_video_error = str(exc)
                logger.warning("YOLO+LPRNet 未加载: %s", exc)
        return self._yolo_video

    def model_available(self) -> bool:
        return self._get_rpnet() is not None or self._get_yolo_video() is not None

    def _ensure_cv2(self) -> None:
        if cv2 is None:
            raise RuntimeError(
                "OpenCV 无法加载，通常是因为当前环境中的 numpy 版本与 opencv-python 不兼容。"
                "请安装兼容版本，例如 `pip install 'numpy<2' --upgrade --force-reinstall`，"
                "或重装匹配当前 Python 版本的 `opencv-python-headless`。"
            ) from _CV2_IMPORT_ERROR

    def _annotate_plates(self, image: np.ndarray, plates: list[dict[str, Any]]) -> np.ndarray:
        self._ensure_cv2()
        annotated = image.copy()
        for plate in plates:
            x1, y1, x2, y2 = plate["bbox"]
            plate_color = plate.get("plate_color", "蓝牌")
            color = ANNOTATE_COLORS.get(plate_color, (0, 255, 255))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{plate['plate_number']} ({plate_color})"
            annotated = draw_cn_text_bgr(
                annotated, label, (x1, max(y1 - 8, 24)), color, font_size=22,
            )
        return annotated

    def _recognize_model(self, image: np.ndarray, filename: str = "") -> list[dict[str, Any]] | None:
        pipeline = self._get_yolo_video()
        if pipeline is None:
            return None

        plates, _ = pipeline.process_frame(image)
        if not plates:
            return None

        results: list[dict[str, Any]] = []
        for plate in plates:
            plate_color = resolve_plate_color(image, plate["bbox"], filename)
            results.append({
                "plate_number": plate.get("plate_number", ""),
                "plate_color": plate_color,
                "bbox": plate.get("bbox", [0, 0, 0, 0]),
                "indices": plate.get("indices", []),
                "confidence": float(plate.get("confidence", 0.85)),
                "source": "yolo_lprnet",
            })
        results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return results

    def _load_image(
        self,
        image_bytes: bytes | None = None,
        img_path: str | None = None,
    ) -> np.ndarray:
        self._ensure_cv2()
        if img_path:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"无法读取图像: {img_path}")
            return image
        if not image_bytes:
            raise ValueError("缺少图像数据")
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法解析图像")
        return image

    def recognize(
        self,
        image_bytes: bytes | None = None,
        filename: str = "",
        img_path: str | None = None,
        force_model: bool = False,
    ) -> dict[str, Any]:
        image = self._load_image(image_bytes=image_bytes, img_path=img_path)
        ccpd_info = parse_ccpd_filename(filename) if filename and not force_model else None
        plates: list[dict] = []
        source = "none"

        if ccpd_info:
            plate_color = resolve_plate_color(image, ccpd_info["bbox"], filename)
            plates.append({
                "plate_number": ccpd_info["plate_number"],
                "plate_color": plate_color,
                "bbox": ccpd_info["bbox"],
                "indices": ccpd_info["indices"],
                "confidence": 1.0,
                "source": "ccpd_gt",
            })
            source = "ccpd_gt"
        else:
            model_plates = self._recognize_model(image, filename)
            if model_plates:
                plates.extend(model_plates)
                source = model_plates[0].get("source", "yolo_lprnet")

        annotated = self._annotate_plates(image, plates) if plates else image.copy()
        return {
            "plates": plates,
            "plate_count": len(plates),
            "annotated_image": ndarray_to_base64(annotated),
            "success": len(plates) > 0,
            "source": source,
            "model_available": self.model_available(),
        }


lpr_service = LicensePlateService()

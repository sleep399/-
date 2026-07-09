"""视频/实时流车牌识别，直接调用 `yolo_lprnet_assets.runtime_api`。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import settings
from app.utils.helpers import ndarray_to_base64

logger = logging.getLogger(__name__)
ASSET_ROOT = (settings.base_dir / "yolo_lprnet_assets").resolve()
if str(ASSET_ROOT) not in sys.path:
    sys.path.insert(0, str(ASSET_ROOT))


class LprVideoService:
    """把后端输入封装成 `yolo_lprnet_assets` 的输入，再把结果转回后端格式。"""

    def __init__(self) -> None:
        self._error: str | None = None
        self._runtime = None
        self._yolo_path: str | None = None
        self._lpr_path: str | None = None

    def _resolve_weights(self) -> tuple[str, str]:
        yolo_candidates = [ASSET_ROOT / "weights" / "best.pt", ASSET_ROOT / "weights" / "yolo11n.pt"]
        lpr_candidates = [ASSET_ROOT / "weights" / "Final_LPRNet_model.pth", ASSET_ROOT / "weights" / "lprnet.pth"]
        yolo = next((p for p in yolo_candidates if p.exists()), None)
        lpr = next((p for p in lpr_candidates if p.exists()), None)
        if not yolo:
            raise FileNotFoundError(f"未找到 YOLO 权重: {ASSET_ROOT / 'weights'}")
        if not lpr:
            raise FileNotFoundError(f"未找到 LPRNet 权重: {ASSET_ROOT / 'weights'}")
        return str(yolo), str(lpr)

    def _load_runtime(self):
        if self._runtime is not None and self._error is None:
            return
        if self._error is not None:
            return
        try:
            from runtime_api import YoloLprRuntime, YoloLprConfig
            yolo_path, lpr_path = self._resolve_weights()
            self._yolo_path, self._lpr_path = yolo_path, lpr_path
            self._runtime = YoloLprRuntime(YoloLprConfig(yolo_model=yolo_path, lpr_model=lpr_path))
        except Exception as exc:
            self._error = str(exc)
            logger.exception("加载 yolo_lprnet_assets runtime 失败: %s", exc)

    def model_available(self) -> bool:
        self._load_runtime()
        return self._error is None and self._runtime is not None

    def model_status(self) -> dict[str, Any]:
        if self.model_available():
            return {
                "model_available": True,
                "engine": "yolo_lprnet",
                "yolo_path": self._yolo_path,
                "lpr_path": self._lpr_path,
                "message": "YOLO+LPRNet 视频识别已就绪",
            }
        return {
            "model_available": False,
            "engine": "yolo_lprnet",
            "message": self._error or f"请将权重放到 `{ASSET_ROOT / 'weights'}`",
        }

    def recognize_frame(self, frame: np.ndarray, frame_index: int = 0) -> dict[str, Any]:
        self._load_runtime()
        if not self.model_available():
            return {
                "plates": [],
                "plate_count": 0,
                "annotated_image": ndarray_to_base64(frame),
                "success": False,
                "source": "yolo_lprnet",
                "model_available": False,
                "frame": frame_index,
                "error": self._error,
            }
        logger.info("[LPR-FRAME] start frame=%s shape=%s", frame_index, getattr(frame, "shape", None))
        result_frame, plate_results = self._runtime.process_frame(frame)
        result = {
            "plates": [
                {
                    "plate_number": p.get("text", "无法识别"),
                    "plate_color": "蓝牌",
                    "bbox": list(p.get("coords", (0, 0, 0, 0))),
                    "indices": [],
                    "confidence": float(p.get("confidence", 0.0)),
                    "source": "yolo_lprnet",
                }
                for p in plate_results
            ],
            "plate_count": len(plate_results),
            "annotated_image": ndarray_to_base64(result_frame),
            "success": len(plate_results) > 0,
            "source": "yolo_lprnet",
            "model_available": True,
            "frame": frame_index,
            "message": "YOLO+LPRNet 视频实时识别",
        }
        logger.info("[LPR-FRAME] done frame=%s plates=%s", frame_index, result.get("plate_count"))
        return result

    def recognize_bytes(self, image_bytes: bytes, frame_index: int = 0) -> dict[str, Any]:
        frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("无法解析视频帧")
        return self.recognize_frame(frame, frame_index)

    def process_video(self, video_path: Path, sample_interval: int = 5) -> dict[str, Any]:
        self._load_runtime()
        logger.info("[LPR-VIDEO] process_video start path=%s interval=%s runtime=%s error=%s", video_path, sample_interval, bool(self._runtime), self._error)
        if not self.model_available():
            logger.error("[LPR-VIDEO] runtime unavailable error=%s", self._error)
            raise RuntimeError(self._error or "YOLO+LPRNet 模型未加载")

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        logger.info("[LPR-VIDEO] video opened fps=%s total_frames=%s", fps, total_frames)

        video_results = self._runtime.process_video_path(str(video_path), sample_interval=max(2, sample_interval))
        logger.info("[LPR-VIDEO] runtime returned frames=%s", len(video_results))
        results: list[dict[str, Any]] = []
        annotated_paths: list[Path] = []
        for item in video_results:
            result_frame = item.get("result_frame")
            plates = item.get("plates", [])
            frame_idx = int(item.get("frame_index", 0))
            frame_path = video_path.parent / f"{video_path.stem}_annotated_{frame_idx:06d}.jpg"
            if result_frame is not None:
                ok = cv2.imwrite(str(frame_path), result_frame)
                logger.info("[LPR-VIDEO] frame saved idx=%s path=%s ok=%s", frame_idx, frame_path, ok)
                annotated_paths.append(frame_path)
            results.append({
                "frame_index": frame_idx,
                "plates": [
                    {
                        "plate_number": p.get("text", "无法识别"),
                        "plate_color": "蓝牌",
                        "bbox": list(p.get("coords", (0, 0, 0, 0))),
                        "indices": [],
                        "confidence": float(p.get("confidence", 0.0)),
                        "source": "yolo_lprnet",
                    }
                    for p in plates
                ],
                "plate_count": len(plates),
                "annotated_image": ndarray_to_base64(result_frame) if result_frame is not None else None,
                "success": len(plates) > 0,
                "source": "yolo_lprnet",
                "model_available": True,
            })

        best = max(results, key=lambda item: sum(p.get("confidence", 0) for p in item.get("plates", []))) if results else None
        annotated_video_path = None
        if annotated_paths:
            first_frame = cv2.imread(str(annotated_paths[0]))
            if first_frame is not None:
                h, w = first_frame.shape[:2]
                annotated_video_path = video_path.parent / f"{video_path.stem}_annotated.mp4"
                writer = cv2.VideoWriter(str(annotated_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                logger.info("[LPR-VIDEO] writing annotated video path=%s size=%sx%s frames=%s", annotated_video_path, w, h, len(annotated_paths))
                for p in annotated_paths:
                    img = cv2.imread(str(p))
                    if img is not None:
                        writer.write(img)
                writer.release()
                logger.info("[LPR-VIDEO] annotated video written exists=%s", annotated_video_path.exists())
        logger.info("[LPR-VIDEO] process_video done best=%s annotated=%s", bool(best), annotated_video_path)
        return {
            "frame_count": len(results),
            "total_frames": len(video_results),
            "results": results,
            "best": best,
            "annotated_frames": [str(p) for p in annotated_paths],
            "annotated_video_path": str(annotated_video_path) if annotated_video_path else None,
            "model_available": True,
        }


lpr_video_service = LprVideoService()

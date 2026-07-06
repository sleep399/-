import io
import math
import time
from collections import Counter, deque
from typing import Any
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageSequence
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from app.config import settings
from app.utils.helpers import ndarray_to_base64
from app.utils.model_loader import get_model_path


OWNER_GESTURES = {
    "no_gesture": ("no_gesture", "无手势", None),
    "palm_open": ("palm_open", "手掌张开", "wake"),
    "fist": ("fist", "握拳", "confirm"),
    "circle": ("circle", "单指画圈", "volume_adjust"),
    "swipe_left": ("swipe_left", "左滑", "prev_page"),
    "swipe_right": ("swipe_right", "右滑", "next_page"),
    "thumb_up": ("thumb_up", "拇指向上", "answer_call"),
    "thumb_down": ("thumb_down", "拇指向下", "hang_up"),
    "wave": ("wave", "挥手", "go_home"),
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17),
]


class OwnerGestureService:
    def __init__(self):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=get_model_path("hand_landmarker.task")),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._position_history: deque = deque(maxlen=20)
        self._gesture_start: dict[str, float] = {}
        self._circle_points: deque = deque(maxlen=30)
        self._last_gesture: str = "no_gesture"
        self._last_action_time: float = 0

    def _distance(self, a, b) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _classify(self, landmarks, w, h) -> tuple[str, float]:
        lm = landmarks
        index_tip = (lm[8].x * w, lm[8].y * h)

        fingers_up = sum([lm[8].y < lm[6].y, lm[12].y < lm[10].y, lm[16].y < lm[14].y, lm[20].y < lm[18].y])
        thumb_up = lm[4].y < lm[3].y

        self._position_history.append(index_tip)
        self._circle_points.append(index_tip)

        if fingers_up >= 4 and thumb_up:
            return "palm_open", 0.9
        if fingers_up == 0 and not thumb_up:
            return "fist", 0.85
        if thumb_up and fingers_up == 0:
            return "thumb_up", 0.8
        if lm[4].y > lm[3].y and fingers_up == 0:
            return "thumb_down", 0.75

        if len(self._position_history) >= 8:
            xs = [p[0] for p in self._position_history]
            dx = xs[-1] - xs[0]
            if dx < -60:
                return "swipe_left", 0.8
            if dx > 60:
                return "swipe_right", 0.8

        if len(self._circle_points) >= 15:
            cx = sum(p[0] for p in self._circle_points) / len(self._circle_points)
            cy = sum(p[1] for p in self._circle_points) / len(self._circle_points)
            dists = [self._distance(p, (cx, cy)) for p in self._circle_points]
            if max(dists) - min(dists) < 40 and max(dists) > 30:
                return "circle", 0.75

        if len(self._position_history) >= 5:
            ys = [p[1] for p in list(self._position_history)[-5:]]
            if max(ys) - min(ys) > 50 and fingers_up >= 3:
                return "wave", 0.7

        return "no_gesture", 0.3

    def _apply_debounce(self, gesture: str, confidence: float) -> tuple[str, float, str | None]:
        now = time.time()
        if gesture == "no_gesture":
            self._gesture_start.clear()
            return gesture, confidence, None

        if gesture not in self._gesture_start:
            self._gesture_start = {gesture: now}
            return "no_gesture", confidence, None

        held = now - self._gesture_start.get(gesture, now)
        if held < settings.gesture_hold_threshold:
            return "no_gesture", confidence, None

        if now - self._last_action_time < 1.0 and gesture == self._last_gesture:
            return "no_gesture", confidence, None

        _, _, action = OWNER_GESTURES.get(gesture, ("no_gesture", "无手势", None))
        self._last_gesture = gesture
        self._last_action_time = now
        self._gesture_start.clear()
        return gesture, confidence, action

    def _draw_hand(self, image, landmarks, w, h):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(image, pts[a], pts[b], (255, 128, 0), 2)
        for p in pts:
            cv2.circle(image, p, 4, (0, 200, 255), -1)

    def _detect_best_frame(self, image_bytes: bytes) -> np.ndarray:
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            if getattr(pil_img, "is_animated", False):
                best_frame = None
                best_score = -1.0
                for frame in ImageSequence.Iterator(pil_img):
                    frame_rgb = frame.convert("RGB")
                    frame_np = cv2.cvtColor(np.array(frame_rgb), cv2.COLOR_RGB2BGR)
                    score = cv2.Laplacian(cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                    if score > best_score:
                        best_score = score
                        best_frame = frame_np
                if best_frame is not None:
                    return best_frame
            frame_rgb = pil_img.convert("RGB")
            return cv2.cvtColor(np.array(frame_rgb), cv2.COLOR_RGB2BGR)
        except Exception:
            pass
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法解析图像")
        return image

    def recognize(self, image_bytes: bytes) -> dict[str, Any]:
        image = self._detect_best_frame(image_bytes)

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        annotated = image.copy()

        gesture, confidence = "no_gesture", 0.0
        action = None
        keypoints = []

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            self._draw_hand(annotated, hand, w, h)
            gesture, confidence = self._classify(hand, w, h)
            keypoints = [{"id": i, "x": round(lm.x * w, 2), "y": round(lm.y * h, 2), "z": round(lm.z, 4)} for i, lm in enumerate(hand)]
            gesture, confidence, action = self._apply_debounce(gesture, confidence)
            if gesture == "no_gesture" and confidence < 0.5:
                gesture = "palm_open"
                confidence = 0.6

        en, cn, _ = OWNER_GESTURES.get(gesture, OWNER_GESTURES["no_gesture"])
        color = (0, 200, 255) if action else (180, 180, 180)
        if gesture != "no_gesture":
            cv2.putText(annotated, f"{cn} ({confidence:.0%})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            if action:
                cv2.putText(annotated, f"Action: {action}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return {
            "gesture": en,
            "gesture_cn": cn,
            "confidence": round(confidence, 3),
            "action": action,
            "keypoints": keypoints,
            "annotated_image": ndarray_to_base64(annotated),
            "success": action is not None or gesture != "no_gesture",
        }

    def recognize_frame(self, frame: np.ndarray) -> dict[str, Any]:
        _, buf = cv2.imencode(".jpg", frame)
        return self.recognize(buf.tobytes())

    def apply_action_to_state(self, action: str, state: dict) -> dict:
        if not action:
            return state
        if action == "wake":
            state["is_awake"] = 1
        elif action == "volume_adjust":
            state["volume"] = min(100, state.get("volume", 50) + 5)
        elif action == "prev_page":
            pages = ["home", "media", "climate", "phone"]
            idx = pages.index(state.get("current_page", "home"))
            state["current_page"] = pages[max(0, idx - 1)]
        elif action == "next_page":
            pages = ["home", "media", "climate", "phone"]
            idx = pages.index(state.get("current_page", "home"))
            state["current_page"] = pages[min(len(pages) - 1, idx + 1)]
        elif action == "answer_call":
            state["phone_status"] = "in_call"
        elif action == "hang_up":
            state["phone_status"] = "idle"
        elif action == "go_home":
            state["current_page"] = "home"
        return state


owner_gesture_service = OwnerGestureService()

from __future__ import annotations
from datetime import datetime
from typing import Any


class AlertAgent:
    def __init__(self):
        self._sockets = set()
        self._confidence = {"police": []}

    def register_ws(self, websocket):
        self._sockets.add(websocket)

    def unregister_ws(self, websocket):
        self._sockets.discard(websocket)

    def record_gesture_confidence(self, module: str, confidence: float):
        values = self._confidence.setdefault(module, [])
        values.append(float(confidence))
        del values[:-20]

    async def check_and_alert(self, db, module: str):
        return None

    async def trigger_alert(self, db, event_type: str, level: str, context: dict[str, Any]):
        return type("Alert", (), {"id": 0, "title": event_type, "summary": str(context)})()

    def get_stats(self, db):
        return {"total": 0, "by_level": {}, "recent": []}


alert_agent = AlertAgent()

from __future__ import annotations
import json
from sqlalchemy.orm import Session
from app.models.logs import SystemLog


def write_log(db: Session, category: str, message: str, level: str = "INFO", detail: dict | None = None, user_id: int | None = None):
    try:
        log = SystemLog(category=category, level=level, message=message, detail=json.dumps(detail or {}, ensure_ascii=False), user_id=user_id)
        db.add(log)
        db.commit()
        return log
    except Exception:
        db.rollback()
        return None

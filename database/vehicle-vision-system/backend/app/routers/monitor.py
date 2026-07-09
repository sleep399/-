from __future__ import annotations
from datetime import datetime
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.models.alerts import AlertEvent
from app.models.logs import SystemLog
from app.schemas import AlertResponse, LogResponse
from app.services.alert_agent import alert_agent

router = APIRouter(prefix="/api/monitor", tags=["monitor"])


@router.get("/logs", response_model=List[LogResponse], summary="System logs")
def get_logs(
    category: str | None = None,
    level: str | None = None,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    q = db.query(SystemLog).order_by(SystemLog.created_at.desc())
    if category:
        q = q.filter(SystemLog.category == category)
    if level:
        q = q.filter(SystemLog.level == level)
    return q.offset(skip).limit(limit).all()


@router.get("/alerts", response_model=List[AlertResponse], summary="Alert history")
def get_alerts(
    level: str | None = None,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    q = db.query(AlertEvent).order_by(AlertEvent.created_at.desc())
    if level:
        q = q.filter(AlertEvent.level == level)
    return q.offset(skip).limit(limit).all()


@router.get("/alerts/stats", summary="Alert stats")
def alert_stats(db: Session = Depends(get_db)):
    return alert_agent.get_stats(db)


@router.post("/alerts/{alert_id}/resolve", summary="Resolve alert")
def resolve_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = db.query(AlertEvent).get(alert_id)
    if not alert:
        return {"message": "alert not found"}`r`n    alert.status = "resolved"
    alert.resolved_at = datetime.utcnow()
    db.commit()
    return {"message": "resolved", "id": alert_id}`r`n`r`n`r`n@router.post("/alerts/test", summary="瑙﹀彂娴嬭瘯鍛婅")
async def test_alert(db: Session = Depends(get_db)):
    alert = await alert_agent.trigger_alert(db, "test_event", "info", {"source": "manual_test"})
    return {"id": alert.id, "title": alert.title, "summary": alert.summary}


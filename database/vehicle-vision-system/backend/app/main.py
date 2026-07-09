from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.config import settings
from app.database import SessionLocal, init_db
from app.models.user import User
from app.utils.auth import hash_password
from app.routers import auth, police_gesture, websocket


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            db.add(User(username="admin", email="admin@demo.com", hashed_password=hash_password("admin123"), is_active=True))
            db.commit()
    finally:
        db.close()
    yield


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan, docs_url="/api/docs", redoc_url="/api/redoc", openapi_url="/api/openapi.json")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(police_gesture.router)
app.include_router(websocket.router)


@app.get("/api/lpr/history")
def lpr_history(limit: int = 100):
    return []


@app.get("/api/owner-gesture/history")
def owner_history(limit: int = 100):
    return []


@app.get("/api/owner-gesture/gestures")
def owner_gestures():
    return []


@app.get("/api/owner-gesture/vehicle-state")
def vehicle_state():
    return {"volume": 50, "temperature": 24, "phone_status": "idle", "current_page": "home", "is_awake": 1}


@app.get("/api/monitor/alerts/stats")
def alert_stats():
    return {"total": 0, "by_level": {}, "recent": []}


@app.get("/api/monitor/alerts")
def alerts(limit: int = 30):
    return []


@app.get("/api/monitor/logs")
def logs(limit: int = 50):
    return []


@app.post("/api/monitor/alerts/test")
def test_alert():
    return {"id": 0, "title": "Test alert", "summary": "Alert test is available."}


static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": settings.app_name, "docs": "/api/docs"}

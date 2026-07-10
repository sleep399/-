from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from app.config import settings
from app.database import init_db, SessionLocal, check_db_connection
from app.models.user import User
from app.database import SessionLocal
from app.utils.auth import hash_password
from app.utils.logger import get_logger, write_system_log, write_log
from app.routers import auth, lpr, police_gesture, owner_gesture, monitor, websocket
from app.services.alert_agent import alert_agent
from app.services.llm_service import llm_service

main_logger = get_logger("main")


async def _startup_checks(db):
    """启动时配置校验与系统日志"""
    write_system_log(db, "系统启动完成", level="INFO", detail={"version": "1.0.0"})

    if not check_db_connection():
        alert_agent.record_db_connection(False)
        await alert_agent.check_and_alert(db, "db")
        write_system_log(db, "数据库连接检查失败", level="ERROR")
    else:
        alert_agent.record_db_connection(True)
        write_system_log(db, "数据库连接正常", level="INFO")

    if settings.alert_webhook_enabled and not settings.webhook_url:
        await alert_agent.handle_config_missing(db, "webhook_url", severity="warning")
        write_system_log(db, "Webhook 告警已启用但未配置 URL", level="WARN")

    if settings.alert_email_enabled and not all([settings.smtp_host, settings.smtp_user, settings.alert_email_to]):
        await alert_agent.handle_config_missing(db, "smtp/email", severity="warning")
        write_system_log(db, "邮件告警已启用但 SMTP 配置不完整", level="WARN")

    if not settings.llm_configured:
        write_system_log(db, "LLM API Key 未配置，告警摘要将使用模板降级", level="WARN")
    else:
        llm_status = await llm_service.test_connection()
        if llm_status.get("ok"):
            write_system_log(
                db, "LLM API 连接正常",
                level="INFO",
                detail={
                    "provider": llm_status.get("provider"),
                    "model": llm_status.get("model"),
                },
            )
            main_logger.info(
                "LLM 已就绪: %s / %s",
                llm_status.get("provider_label"),
                llm_status.get("model"),
            )
        else:
            write_system_log(
                db, "LLM API 连接失败，告警摘要将使用模板降级",
                level="WARN",
                detail={"error": llm_status.get("error") or llm_status.get("message")},
            )
            await alert_agent.check_and_alert(db, "llm")
            main_logger.warning("LLM 连接失败: %s", llm_status.get("message"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动初始化
    main_logger.info("系统启动中...")
    init_db()

    db = SessionLocal()
    try:
        try:
            admin = db.query(User).filter(User.username == "admin").first()
        except Exception:
            admin = None
        if not admin:
            db.execute(
                User.__table__.insert().values(
                    username="admin",
                    email="admin@demo.com",
                    hashed_password=hash_password("admin123"),
                    is_active=True,
                )
            )
            db.commit()
            main_logger.info("默认管理员账号已创建: admin/admin123")
            write_system_log(db, "默认管理员账号已创建", level="INFO", detail={"username": "admin"})

        await _startup_checks(db)
    finally:
        db.close()

    await alert_agent.start_patrol_loop(SessionLocal)
    main_logger.info("告警智能体后台巡检已启动")

    yield

    # 关闭清理
    main_logger.info("系统关闭中...")
    db = SessionLocal()
    try:
        write_system_log(db, "系统正在关闭", level="INFO")
    finally:
        db.close()


app = FastAPI(
    title=settings.app_name,
    description="车载摄像头视觉感知与人机交互系统 - 车牌识别、交警手势、车主手势控车、告警智能体",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(lpr.router)
app.include_router(police_gesture.router)
app.include_router(owner_gesture.router)
app.include_router(monitor.router)
app.include_router(websocket.router)

static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(
            str(index_file),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return {"message": "车载视觉感知系统 API", "docs": "/api/docs"}


@app.get("/api/docs", include_in_schema=False)
async def api_docs():
    """自定义 Anthropic 风格 Swagger UI"""
    docs_file = static_dir / "docs.html"
    if docs_file.exists():
        return FileResponse(str(docs_file))
    return {"message": "API 文档未找到"}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    response = await call_next(request)
    if response.status_code in {401, 403} and request.url.path.startswith("/api/"):
        db = SessionLocal()
        try:
            client_ip = (
                request.headers.get("x-forwarded-for", "")
                .split(",")[0]
                .strip()
                or (request.client.host if request.client else "unknown")
            )
            await alert_agent.handle_unauthorized_access(
                db,
                request.url.path,
                ip=client_ip,
                user_agent=request.headers.get("user-agent"),
            )
            write_log(
                db,
                "user",
                f"未授权访问: {request.url.path}",
                level="WARN",
                detail={
                    "ip": client_ip,
                    "path": request.url.path,
                    "status": response.status_code,
                    "user_agent": request.headers.get("user-agent"),
                },
            )
        except Exception as e:
            main_logger.warning(f"未授权访问检测失败: {e}")
        finally:
            db.close()
    return response

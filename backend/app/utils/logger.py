import json
from datetime import datetime
from sqlalchemy.orm import Session
from app.models.logs import SystemLog
from app.services.log_stream import broadcast_log

# ──────────────────────────────────────────────
# 本地时区（中国标准时间 UTC+8）—— 公共工具函数
# ──────────────────────────────────────────────

_TZ_CN = timezone(timedelta(hours=8))


def localize_utc(dt: Optional[datetime]) -> Optional[str]:
    """将 naive UTC datetime 转为本地时间 (UTC+8) ISO 字符串。

    用法: localize_utc(record.created_at)
    数据库中存储的是 datetime.utcnow() 的 naive datetime，
    此函数将其当作 UTC 转换为北京时间后返回 ISO 格式字符串。
    """
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(_TZ_CN).isoformat()
    return dt.replace(tzinfo=timezone.utc).astimezone(_TZ_CN).isoformat()


def local_now_cn(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """返回当前北京时间字符串（用于 LLM 提示等）。"""
    return datetime.now(_TZ_CN).strftime(fmt)


# ──────────────────────────────────────────────
# Python logging 体系 – 文件轮转 + 控制台输出
# ──────────────────────────────────────────────

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

_py_logger = logging.getLogger("vehicle-vision")
_py_logger.setLevel(logging.DEBUG)

# 避免重复添加 handler（热重载场景）
if not _py_logger.handlers:
    # 文件轮转 handler —— 单文件最大 20MB，保留 10 个备份
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(LOG_DIR, "app.log"),
        maxBytes=20 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    _py_logger.addHandler(file_handler)

    # 告警专用日志文件
    alert_file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(LOG_DIR, "alerts.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    alert_file_handler.setLevel(logging.WARNING)
    alert_file_handler.setFormatter(file_fmt)
    alert_file_handler.addFilter(lambda record: record.levelno >= logging.WARNING)
    _py_logger.addHandler(alert_file_handler)

    # 错误专用日志文件
    error_file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(LOG_DIR, "errors.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(file_fmt)
    _py_logger.addHandler(error_file_handler)

    # 控制台 handler（仅 debug 模式下）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "[%(levelname)s] %(name)s - %(message)s"
    )
    console_handler.setFormatter(console_fmt)
    _py_logger.addHandler(console_handler)


def get_logger(name: str | None = None) -> logging.Logger:
    """获取命名 logger"""
    if name:
        return _py_logger.getChild(name)
    return _py_logger


def _level_std_to_py(level: str) -> int:
    """DB 级别名称 -> Python logging 级别"""
    return {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARN": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}.get(level.upper(), logging.INFO)


_LEVEL_MAP = {"INFO": "info", "WARN": "warning", "ERROR": "error", "CRITICAL": "critical"}


def write_log(
    db: Session,
    category: str,
    message: str,
    level: str = "INFO",
    detail: dict | None = None,
    user_id: int | None = None,
):
    log = SystemLog(
        category=category,
        level=level,
        message=message,
        detail_json=json.dumps(detail, ensure_ascii=False) if detail else None,
        user_id=user_id,
        created_at=datetime.utcnow(),
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log

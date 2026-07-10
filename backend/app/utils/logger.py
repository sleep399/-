import json
import logging
import logging.handlers
import os
import traceback
from datetime import datetime, timezone, timedelta
from typing import Optional

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
) -> SystemLog:
    """写入数据库日志 + Python logging 双通道。

    日志类别:
      - lpr          车牌识别日志
      - police_gesture 交警手势识别日志
      - owner_gesture  车主手势识别日志
      - alert        告警日志
      - user         用户操作日志
      - system       系统运行日志
      - agent        智能体决策日志
    """
    now = datetime.utcnow()
    detail_json = json.dumps(detail, ensure_ascii=False) if detail else None

    # 1) 数据库持久化
    log = SystemLog(
        category=category,
        level=level,
        message=message,
        detail_json=detail_json,
        user_id=user_id,
        created_at=now,
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    # 2) Python logging 文件/控制台输出
    py_level = _level_std_to_py(level)
    py_method = _LEVEL_MAP.get(level.upper(), "info")
    py_msg = f"[{category}] {message}"
    if detail:
        py_msg += f" | detail={json.dumps(detail, ensure_ascii=False)}"
    if user_id is not None:
        py_msg += f" | user_id={user_id}"
    getattr(_py_logger, py_method)(py_msg)

    detail_obj = detail
    if detail_obj is None and detail_json:
        try:
            detail_obj = json.loads(detail_json)
        except Exception:
            detail_obj = None

    broadcast_log({
        "id": log.id,
        "category": log.category,
        "level": log.level,
        "message": log.message,
        "detail_json": detail_obj,
        "user_id": log.user_id,
        "created_at": localize_utc(log.created_at),
    })

    return log


def log_exception(
    db: Session,
    category: str,
    message: str,
    exc: Exception,
    user_id: int | None = None,
):
    """记录异常日志（附带完整 traceback）"""
    detail = {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }
    write_log(db, category, message, level="ERROR", detail=detail, user_id=user_id)
    _py_logger.error(f"[{category}] {message} | {type(exc).__name__}: {exc}", exc_info=True)


def write_alert_log(
    db: Session,
    alert_id: int,
    level: str,
    title: str,
    event_type: str,
    summary: str,
    channels: str,
):
    """专门的告警日志记录"""
    detail = {
        "alert_id": alert_id,
        "event_type": event_type,
        "channels": channels,
    }
    write_log(
        db,
        "alert",
        f"[{level.upper()}] [{event_type}] {title} — {summary}",
        level=level.upper(),
        detail=detail,
    )
    _py_logger.log(_level_std_to_py(level), f"ALERT [{event_type}] {title} | channels={channels} | {summary}")


def write_agent_log(
    db: Session,
    message: str,
    level: str = "INFO",
    detail: dict | None = None,
) -> SystemLog:
    """智能体决策日志（告警级别判定、冷却抑制、推送决策等）"""
    return write_log(db, "agent", message, level=level, detail=detail)


def write_system_log(
    db: Session,
    message: str,
    level: str = "INFO",
    detail: dict | None = None,
) -> SystemLog:
    """系统运行日志（启动、关闭、健康检查、配置校验等）"""
    return write_log(db, "system", message, level=level, detail=detail)

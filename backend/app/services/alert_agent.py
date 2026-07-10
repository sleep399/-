import asyncio
import json
import smtplib
from collections import defaultdict, deque
from datetime import datetime
from email.mime.text import MIMEText
from typing import Any

import httpx
from sqlalchemy.orm import Session

from app.config import settings
from app.models.alerts import AlertEvent
from app.utils.logger import write_log, log_exception, write_alert_log, write_agent_log, get_logger, localize_utc as _localize_utc

agent_logger = get_logger("alert_agent")


# ──────────────────────────────────────────────
# 事件类型定义
# ──────────────────────────────────────────────
EVENT_TYPES = {
    "lpr_consecutive_failure": "车牌识别连续失败",
    "lpr_high_failure_rate": "车牌识别失败率过高",
    "gesture_low_confidence": "手势识别置信度持续偏低",
    "llm_api_timeout": "LLM API 调用超时",
    "llm_token_exhausted": "LLM Token 配额即将耗尽",
    "llm_token_exceeded": "LLM Token 配额已超额",
    "unauthorized_access": "未授权访问尝试",
    "service_unhealthy": "系统服务健康异常",
    "model_load_failure": "AI 模型加载失败",
    "database_connection_error": "数据库连接异常",
    "webhook_delivery_failure": "Webhook 推送失败",
    "email_delivery_failure": "邮件推送失败",
    "config_missing": "关键配置缺失",
    "test_event": "测试告警",
}

# 可选配置缺失：只记日志，不反复弹告警（webhook/邮件/LLM 未配属于正常演示状态）
OPTIONAL_CONFIG_KEYS = frozenset({
    "webhook_url", "webhook", "smtp", "smtp/email", "email",
    "llm_api_key", "llm",
})

MERGEABLE_EVENT_TYPES = frozenset({
    "gesture_low_confidence",
})

DEFAULT_LEVELS = {
    "lpr_consecutive_failure": "critical",
    "lpr_high_failure_rate": "warning",
    "gesture_low_confidence": "warning",
    "llm_api_timeout": "critical",
    "llm_token_exhausted": "warning",
    "llm_token_exceeded": "critical",
    "unauthorized_access": "warning",
    "service_unhealthy": "critical",
    "model_load_failure": "critical",
    "database_connection_error": "critical",
    "webhook_delivery_failure": "warning",
    "email_delivery_failure": "warning",
    "config_missing": "warning",
    "test_event": "info",
}


class AlertAgent:
    """告警智能体：感知异常、决策级别、生成摘要、推送通知"""

    LEVELS = {"info": 1, "warning": 2, "critical": 3}

    def __init__(self):
        self._failure_counts: defaultdict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        self._confidence_history: defaultdict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        self._ws_clients: set = set()
        self._lock = asyncio.Lock()

    def register_ws(self, ws):
        self._ws_clients.add(ws)

    def unregister_ws(self, ws):
        self._ws_clients.discard(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self._ws_clients:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._ws_clients.discard(ws)

    def record_lpr_result(self, success: bool):
        self._failure_counts["lpr"].append(0 if success else 1)

    def record_gesture_confidence(self, module: str, confidence: float):
        self._confidence_history[module].append(confidence)

    async def check_and_alert(self, db: Session, module: str) -> AlertEvent | None:
        event = None
        if module == "lpr":
            failures = list(self._failure_counts["lpr"])
            if len(failures) >= settings.alert_failure_threshold and sum(failures[-settings.alert_failure_threshold:]) == settings.alert_failure_threshold:
                event = await self._create_alert(db, "lpr_consecutive_failure", "critical", {"count": settings.alert_failure_threshold, "module": "lpr"})
        elif module in ("police", "owner"):
            confs = list(self._confidence_history[module])
            if len(confs) >= 5 and all(c < settings.low_confidence_threshold for c in confs[-5:]):
                event = await self._create_alert(db, "gesture_low_confidence", "warning", {"confidence": sum(confs[-5:]) / 5, "module": module})
        return event

    async def _check_lpr_anomalies(self, db: Session) -> AlertEvent | None:
        """检测车牌识别异常"""
        failures = list(self._failure_counts["lpr"])
        threshold = settings.alert_failure_threshold

        # 1) 连续失败检测
        if len(failures) >= threshold:
            recent = failures[-threshold:]
            if sum(recent) == threshold:
                return await self.monitor(
                    db, "lpr_consecutive_failure", "critical",
                    {"count": threshold, "module": "lpr", "window": f"最近{threshold}次"}
                )

        # 2) 滑动窗口失败率检测
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=settings.alert_window_seconds)
        window_fails = [
            ts for ts, ok in self._failure_timestamps["lpr"]
            if ts >= cutoff and not ok
        ]
        window_total = len([ts for ts, _ in self._failure_timestamps["lpr"] if ts >= cutoff])
        if window_total >= 10:
            rate = len(window_fails) / window_total
            if rate > settings.alert_anomaly_rate_threshold:
                return await self.monitor(
                    db, "lpr_high_failure_rate", "warning",
                    {"rate": f"{rate:.0%}", "fails": len(window_fails), "total": window_total,
                     "window_seconds": settings.alert_window_seconds}
                )

        return None

    async def _check_gesture_anomalies(self, db: Session, module: str) -> AlertEvent | None:
        """检测手势识别置信度异常"""
        confs = list(self._confidence_history[module])
        if len(confs) >= 5:
            recent_5 = confs[-5:]
            if all(c < settings.low_confidence_threshold for c in recent_5):
                avg_conf = sum(recent_5) / 5
                return await self.monitor(
                    db, "gesture_low_confidence", "warning",
                    {"confidence": avg_conf, "module": module, "threshold": settings.low_confidence_threshold}
                )
        return None

    async def _check_llm_anomalies(self, db: Session) -> AlertEvent | None:
        """检测 LLM 异常（Token 用量 + 调用失败）"""
        # Token 用量检测
        used = self._token_usage["used"]
        limit = self._token_usage["limit"]
        ratio = used / limit if limit > 0 else 0

        if ratio >= 1.0:
            return await self.monitor(
                db, "llm_token_exceeded", "critical",
                {"used": used, "limit": limit, "ratio": f"{ratio:.1%}"}
            )
        if ratio >= (settings.alert_token_critical_threshold / max(settings.alert_token_limit, 1)):
            return await self.monitor(
                db, "llm_token_exhausted", "critical",
                {"used": used, "limit": limit, "ratio": f"{ratio:.1%}",
                 "remaining": limit - used}
            )
        if ratio >= (settings.alert_token_warning_threshold / max(settings.alert_token_limit, 1)):
            return await self.monitor(
                db, "llm_token_exhausted", "warning",
                {"used": used, "limit": limit, "ratio": f"{ratio:.1%}",
                 "remaining": limit - used}
            )

        # LLM 调用失败检测
        llm_fails = list(self._failure_counts["llm"])
        if len(llm_fails) >= 3 and sum(llm_fails[-3:]) >= 2:
            return await self.monitor(
                db, "llm_api_timeout", "critical",
                {"fails": sum(llm_fails[-3:]), "window": "最近3次"}
            )

        return None

    async def _check_db_anomalies(self, db: Session) -> AlertEvent | None:
        """检测数据库异常"""
        fails = list(self._failure_counts["db"])
        if len(fails) >= 3 and all(f == 1 for f in fails[-3:]):
            return await self.monitor(
                db, "database_connection_error", "critical",
                {"consecutive_fails": 3}
            )
        return None

    async def _check_webhook_anomalies(self, db: Session) -> AlertEvent | None:
        """检测 Webhook 推送异常"""
        fails = list(self._failure_counts["webhook"])
        if len(fails) >= 5 and sum(fails[-5:]) >= 3:
            return await self.monitor(
                db, "webhook_delivery_failure", "warning",
                {"fails": sum(fails[-5:]), "window": "最近5次"}
            )
        return None

    async def _check_email_anomalies(self, db: Session) -> AlertEvent | None:
        """检测邮件推送异常"""
        fails = list(self._failure_counts["email"])
        if len(fails) >= 5 and sum(fails[-5:]) >= 3:
            return await self.monitor(
                db, "email_delivery_failure", "warning",
                {"fails": sum(fails[-5:]), "window": "最近5次"}
            )
        return None

    @staticmethod
    def _log_categories_for_event(event_type: str) -> list[str]:
        """告警回放时关联的日志类别"""
        if event_type.startswith("lpr"):
            return ["lpr", "agent", "alert"]
        if "gesture" in event_type:
            return ["police_gesture", "owner_gesture", "agent", "alert"]
        if event_type.startswith("llm"):
            return ["agent", "alert", "system"]
        if event_type == "unauthorized_access":
            return ["user", "agent", "alert"]
        return ["agent", "alert", "system"]

    async def run_patrol(self, db: Session) -> None:
        """后台巡检：重检各模块感知状态并写智能体日志"""
        write_agent_log(
            db,
            "智能体后台巡检：重检车牌/手势/LLM/数据库感知状态",
            level="INFO",
            detail={"modules": ["lpr", "police", "owner", "llm", "db"]},
        )
        for module in ("lpr", "police", "owner", "llm", "db"):
            await self.check_and_alert(db, module)

    async def start_patrol_loop(self, db_factory):
        """启动后台巡检循环（默认 60 秒）"""
        async def _loop():
            while True:
                try:
                    db = db_factory()
                    try:
                        await self.run_patrol(db)
                    finally:
                        db.close()
                except Exception as e:
                    agent_logger.error(f"Patrol failed: {e}")
                await asyncio.sleep(60)

        self._patrol_task = asyncio.create_task(_loop())

    def get_recent_agent_logs(self, db: Session, limit: int = 10) -> list[dict]:
        """获取最近智能体决策/推送日志"""
        from app.models.logs import SystemLog

        rows = (
            db.query(SystemLog)
            .filter(SystemLog.category == "agent")
            .order_by(SystemLog.created_at.desc())
            .limit(limit)
            .all()
        )
        result = []
        for row in rows:
            detail = None
            if row.detail_json:
                try:
                    detail = json.loads(row.detail_json)
                except Exception:
                    detail = row.detail_json
            result.append({
                "id": row.id,
                "level": row.level,
                "message": row.message,
                "detail": detail,
                "created_at": _localize_utc(row.created_at),
            })
        return result

    # ════════════════════════════════════════════
    # 决策模块 —— 自主判定告警级别
    # ════════════════════════════════════════════

    def _decide_level(self, event_type: str, level: str, context: dict) -> str:
        """自主决策告警级别（prompt / warning / critical）

        决策依据:
          1. 事件类型的固有严重程度
          2. 上下文中的量化指标（失败次数、使用率等）
          3. 历史告警频率
        """
        # 根据上下文动态升级
        if event_type == "unauthorized_access":
            count = context.get("count", 1)
            if count >= 10:
                return "critical"  # 频繁未授权访问 → 升级为严重
            return "warning"

        if event_type == "lpr_high_failure_rate":
            rate_str = context.get("rate", "0%").replace("%", "")
            try:
                rate = float(rate_str) / 100
            except (ValueError, TypeError):
                rate = 0
            if rate > 0.6:
                return "critical"
            return "warning"

        if event_type == "llm_token_exhausted":
            ratio_str = context.get("ratio", "0%").replace("%", "")
            try:
                ratio = float(ratio_str) / 100
            except (ValueError, TypeError):
                ratio = 0
            if ratio > 0.95:
                return "critical"
            return "warning"

        if event_type == "gesture_low_confidence":
            conf = context.get("confidence", 0)
            if conf < 0.2:
                return "critical"
            return "warning"

        # 使用默认映射
        return DEFAULT_LEVELS.get(event_type, level)

    # ════════════════════════════════════════════
    # 核心告警流程
    # ════════════════════════════════════════════

    async def monitor(
        self,
        db: Session,
        event_type: str,
        level: str = "warning",
        context: dict | None = None,
        *,
        force_template: bool = False,
    ) -> AlertEvent | None:
        """Agent 核心工作流：感知异常 → 决策级别 → 生成摘要 → 推送通知"""
        observed = context or {}

        # 1) 自主决策告警级别
        decision_level = self._decide_level(event_type, level, observed)
        observed["decided_level"] = decision_level
        observed["original_level"] = level
        dedup_key = self._dedup_key(event_type, observed)

        async with self._lock:
            # 2) 合并到已有未处理告警（限定类型，避免 gesture 等刷屏）
            if event_type in MERGEABLE_EVENT_TYPES:
                existing = self._find_mergeable_open_alert(db, event_type, observed)
                if existing:
                    agent_logger.info(f"Alert merged into #{existing.id}: {event_type}")
                    write_agent_log(
                        db,
                        f"告警合并抑制新建: {EVENT_TYPES.get(event_type, event_type)} → #{existing.id}",
                        level="INFO",
                        detail={
                            "event_type": event_type,
                            "merged_into": existing.id,
                            "dedup_key": dedup_key,
                        },
                    )
                    return await self._merge_alert(db, existing, decision_level, observed)

            # 3) 冷却检查（同类型告警间隔）
            if not self._should_alert(event_type, observed):
                agent_logger.info(
                    f"Alert suppressed by cooldown: {dedup_key} "
                    f"(last={self._last_alert_time.get(dedup_key)})"
                )
                write_agent_log(
                    db,
                    f"告警冷却抑制: {EVENT_TYPES.get(event_type, event_type)}",
                    level="INFO",
                    detail={
                        "event_type": event_type,
                        "dedup_key": dedup_key,
                        "decided_level": decision_level,
                        "last_alert_at": _localize_utc(self._last_alert_time.get(dedup_key)),
                    },
                )
                return None

            # 并发保护：先占位冷却，避免多帧同时创建多条告警
            self._last_alert_time[dedup_key] = datetime.utcnow()

        write_agent_log(
            db,
            f"告警级别决策: {EVENT_TYPES.get(event_type, event_type)} → {decision_level}",
            level="INFO" if decision_level == "info" else ("WARN" if decision_level == "warning" else "CRITICAL"),
            detail={
                "event_type": event_type,
                "dedup_key": dedup_key,
                "original_level": level,
                "decided_level": decision_level,
                "context": observed,
            },
        )

        # 4) 生成告警（LLM 较慢，放在锁外）
        return await self.trigger_alert(db, event_type, decision_level, observed, force_template=force_template)

    @staticmethod
    def _parse_detail_json(alert: AlertEvent) -> dict:
        if not alert.detail_json:
            return {}
        try:
            return json.loads(alert.detail_json)
        except Exception:
            return {}

    def _dedup_key(self, event_type: str, context: dict | None = None) -> str:
        ctx = context or {}
        if event_type == "gesture_low_confidence":
            return f"{event_type}:{ctx.get('module', 'unknown')}"
        if event_type == "model_load_failure":
            return f"{event_type}:{ctx.get('model_name', 'unknown')}"
        return event_type

    def _cooldown_seconds(self, event_type: str) -> int:
        if event_type == "gesture_low_confidence":
            return settings.alert_gesture_cooldown_seconds
        if event_type == "config_missing":
            return settings.alert_config_cooldown_seconds
        return settings.alert_cooldown_seconds

    def _find_mergeable_open_alert(
        self, db: Session, event_type: str, context: dict
    ) -> AlertEvent | None:
        target_key = self._dedup_key(event_type, context)
        rows = (
            db.query(AlertEvent)
            .filter(AlertEvent.event_type == event_type, AlertEvent.status == "open")
            .order_by(AlertEvent.created_at.desc())
            .limit(30)
            .all()
        )
        for row in rows:
            if self._dedup_key(event_type, self._parse_detail_json(row)) == target_key:
                return row
        return None

    async def _merge_alert(
        self,
        db: Session,
        alert: AlertEvent,
        level: str,
        context: dict,
    ) -> AlertEvent:
        existing_detail = self._parse_detail_json(alert)
        count = int(existing_detail.get("occurrence_count", 1)) + 1
        merged_context = {**existing_detail, **context}
        merged_context["occurrence_count"] = count
        merged_context["last_occurrence_at"] = datetime.utcnow().isoformat()

        if self.LEVELS.get(level, 0) > self.LEVELS.get(alert.level, 0):
            alert.level = level

        alert.detail_json = json.dumps(merged_context, ensure_ascii=False)

        conf = context.get("confidence")
        suffix = f"（同类异常已累计 {count} 次"
        if isinstance(conf, (int, float)):
            suffix += f"，最近平均置信度 {conf:.0%}"
        suffix += "）"
        base_summary = (alert.summary or "").split("（同类异常已累计")[0].strip()
        if base_summary:
            alert.summary = f"{base_summary}{suffix}"
        else:
            title = alert.title or EVENT_TYPES.get(alert.event_type, alert.event_type)
            alert.summary = f"{title}{suffix}"

        alert.system_health_json = json.dumps({
            "perception": self.get_perception_snapshot(),
            "decision_level": alert.level,
            "event_type": alert.event_type,
            "merged": True,
            "occurrence_count": count,
        }, ensure_ascii=False)
        db.commit()
        db.refresh(alert)

        dedup_key = self._dedup_key(alert.event_type, context)
        self._last_alert_time[dedup_key] = datetime.utcnow()

        write_agent_log(
            db,
            f"告警合并更新 #{alert.id}: {EVENT_TYPES.get(alert.event_type, alert.event_type)} ×{count}",
            level="INFO",
            detail={"alert_id": alert.id, "occurrence_count": count, "dedup_key": dedup_key},
        )
        return alert

    def _should_alert(self, event_type: str, context: dict | None = None) -> bool:
        """检查是否应该发送告警（冷却机制，按去重键区分）。"""
        key = self._dedup_key(event_type, context)
        last = self._last_alert_time.get(key)
        if last is None:
            return True
        cooldown = timedelta(seconds=self._cooldown_seconds(event_type))
        if datetime.utcnow() - last >= cooldown:
            return True
        return False

    async def trigger_alert(
        self,
        db: Session,
        event_type: str,
        level: str = "warning",
        context: dict | None = None,
    ) -> AlertEvent:
        return await self._create_alert(db, event_type, level, context or {})

    async def _create_alert(
        self,
        db: Session,
        event_type: str,
        level: str,
        context: dict,
    ) -> AlertEvent:
        summary_data = await llm_service.generate_alert_summary(event_type, level, context)
        alert = AlertEvent(
            level=level,
            event_type=event_type,
            title=summary_data.get("title", event_type),
            summary=summary_data.get("summary", ""),
            detail_json=json.dumps(context, ensure_ascii=False),
            root_cause=summary_data.get("root_cause"),
            suggestion=summary_data.get("suggestion"),
            channels_sent="web",
            created_at=datetime.utcnow(),
        )
        db.add(alert)
        db.commit()
        db.refresh(alert)

        # 3) 记录告警冷却时间
        self._last_alert_time[self._dedup_key(event_type, context)] = now

        payload = {
            "type": "alert",
            "id": alert.id,
            "level": level,
            "event_type": event_type,
            "title": alert.title,
            "summary": alert.summary,
            "root_cause": alert.root_cause,
            "suggestion": alert.suggestion,
            "created_at": alert.created_at.isoformat(),
        }
        await self.broadcast(payload)

        channels = ["web"]
        if await self._send_webhook(payload):
            channels.append("webhook")
        if await self._send_email(alert):
            channels.append("email")
        alert.channels_sent = ",".join(channels)
        db.commit()

        return alert

    async def _send_webhook(self, payload: dict) -> bool:
        if not settings.webhook_url:
            return False
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(settings.webhook_url, json={
                    "msgtype": "text",
                    "text": {"content": f"[{payload['level'].upper()}] {payload['title']}\n{payload['summary']}\n建议: {payload.get('suggestion', '')}"},
                })
            return True
        except Exception:
            return False

    async def _send_email(self, alert: AlertEvent) -> bool:
        if not all([settings.smtp_host, settings.smtp_user, settings.alert_email_to]):
            return False
        try:
            msg = MIMEText(f"{alert.summary}\n\n根因: {alert.root_cause}\n建议: {alert.suggestion}")
            msg["Subject"] = f"[{alert.level.upper()}] {alert.title}"
            msg["From"] = settings.smtp_user
            msg["To"] = settings.alert_email_to
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
                server.starttls()
                server.login(settings.smtp_user, settings.smtp_password)
                server.send_message(msg)
            return True
        except Exception:
            return False

    async def send_test_notification(self, channel: str = "all") -> dict[str, Any]:
        """向指定渠道发送测试通知，不写入告警库。"""
        from types import SimpleNamespace

        now = datetime.utcnow()
        created_str = _localize_utc(now) or now.isoformat()
        payload = {
            "type": "test",
            "id": 0,
            "level": "info",
            "event_type": "test_event",
            "title": "【测试】车载视觉系统通知渠道连通性检查",
            "summary": "这是一条测试消息，用于验证 Webhook / 邮件等外部通知渠道配置是否正确。",
            "root_cause": "用户手动触发通知测试",
            "suggestion": "若收到本消息，说明对应渠道配置正常",
            "detail": {"source": "notification_test"},
            "created_at": created_str,
        }
        results: dict[str, Any] = {"channel": channel, "channels": {}}

        if channel in ("web", "all"):
            await self.broadcast(payload)
            results["channels"]["web"] = {"ok": True}
            if settings.alert_sse_enabled:
                await self.broadcast_sse(payload)
                results["channels"]["sse"] = {"ok": True}
            else:
                results["channels"]["sse"] = {"ok": False, "reason": "未启用"}

        if channel in ("webhook", "all"):
            if not settings.alert_webhook_enabled:
                results["channels"]["webhook"] = {"ok": False, "reason": "未启用"}
            elif not settings.webhook_url:
                results["channels"]["webhook"] = {"ok": False, "reason": "未配置 URL"}
            else:
                ok = await self._send_webhook(payload)
                results["channels"]["webhook"] = {"ok": ok}

        if channel in ("email", "all"):
            if not settings.alert_email_enabled:
                results["channels"]["email"] = {"ok": False, "reason": "未启用"}
            elif not all([settings.smtp_host, settings.smtp_user, settings.alert_email_to]):
                results["channels"]["email"] = {"ok": False, "reason": "SMTP 配置不完整"}
            else:
                fake_alert = SimpleNamespace(
                    level="info",
                    title=payload["title"],
                    event_type="test_event",
                    summary=payload["summary"],
                    root_cause=payload["root_cause"],
                    suggestion=payload["suggestion"],
                    created_at=now,
                )
                ok = await self._send_email(fake_alert)
                results["channels"]["email"] = {"ok": ok}

        return results

    # ════════════════════════════════════════════
    # 统计与可视化数据
    # ════════════════════════════════════════════

    def _alert_to_dict(self, a: AlertEvent) -> dict[str, Any]:
        """将告警 ORM 对象转为 API 字典"""
        detail = {}
        if a.detail_json:
            try:
                detail = json.loads(a.detail_json)
            except Exception:
                detail = {"raw": a.detail_json}
        return {
            "id": a.id,
            "level": a.level,
            "event_type": a.event_type,
            "event_type_cn": EVENT_TYPES.get(a.event_type, a.event_type),
            "title": a.title,
            "summary": a.summary,
            "root_cause": a.root_cause,
            "suggestion": a.suggestion,
            "channels": a.channels_sent,
            "status": a.status,
            "resolution_note": a.resolution_note,
            "detail": detail,
            "system_health": json.loads(a.system_health_json) if a.system_health_json else {},
            "created_at": _localize_utc(a.created_at),
            "resolved_at": _localize_utc(a.resolved_at),
        }

    def _compute_mttr_minutes(self, db: Session) -> float | None:
        """计算平均处理时长（分钟）"""
        resolved = (
            db.query(AlertEvent)
            .filter(AlertEvent.status == "resolved", AlertEvent.resolved_at.isnot(None))
            .all()
        )
        if not resolved:
            return None
        total_minutes = 0.0
        count = 0
        for a in resolved:
            if a.created_at and a.resolved_at:
                delta = (a.resolved_at - a.created_at).total_seconds() / 60
                if delta >= 0:
                    total_minutes += delta
                    count += 1
        return round(total_minutes / count, 1) if count else None

    def consolidate_duplicate_open_alerts(self, db: Session) -> int:
        """同模块 gesture_low_confidence 仅保留最新一条 open，其余标记已处理。"""
        open_rows = (
            db.query(AlertEvent)
            .filter(AlertEvent.status == "open", AlertEvent.event_type == "gesture_low_confidence")
            .order_by(AlertEvent.created_at.desc())
            .all()
        )
        seen_keys: set[str] = set()
        resolved = 0
        now = datetime.utcnow()
        for row in open_rows:
            key = self._dedup_key(row.event_type, self._parse_detail_json(row))
            if key in seen_keys:
                row.status = "resolved"
                row.resolved_at = now
                row.resolution_note = "系统自动合并：同类型重复手势告警"
                resolved += 1
            else:
                seen_keys.add(key)
        if resolved:
            db.commit()
        return resolved

    def get_stats(self, db: Session) -> dict[str, Any]:
        alerts = db.query(AlertEvent).order_by(AlertEvent.created_at.desc()).limit(100).all()
        by_level = defaultdict(int)
        by_type = defaultdict(int)
        for a in alerts:
            by_level[a.level] += 1
            by_type[a.event_type] += 1
        return {
            "total": len(alerts),
            "by_level": dict(by_level),
            "by_type": dict(by_type),
            "recent": [
                {
                    "id": a.id,
                    "level": a.level,
                    "event_type": a.event_type,
                    "title": a.title,
                    "summary": a.summary,
                    "status": a.status,
                    "created_at": a.created_at.isoformat(),
                }
                for a in alerts[:20]
            ],
        }


alert_agent = AlertAgent()

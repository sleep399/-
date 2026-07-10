"""LLM 服务 —— 调用预训练大模型 API 生成告警摘要与智能助手回答（OpenAI 兼容接口）。"""

import json
import re
from datetime import datetime
from typing import Any

import httpx

from app.config import settings, LLM_PROVIDER_PRESETS
from app.database import SessionLocal
from app.utils.logger import get_logger
from app.utils.user_language import (
    event_type_to_user,
    level_to_user,
    assistant_answer_for_user,
    build_assistant_knowledge,
    detect_assistant_intent,
    humanize_tech_terms,
    _get_plan,
    _format_steps,
    _format_steps_conversational,
    _is_useless_suggestion,
)

llm_logger = get_logger("llm_service")

ALERT_SUMMARY_SYSTEM = """你是车载视觉感知系统的告警助手「小智」，正在帮车主/管理员解读系统异常。

说话风格：像一位靠谱同事在口头汇报——自然、有温度、好懂，不要公文腔或机器人腔。
禁止：「经检测」「请注意」「处理方法：」「影响范围：」等模板化标题；禁止 API 路径、Token、Webhook 等英文术语。

输出要求：
1. 必须返回合法 JSON，不要 Markdown 代码块
2. 各字段用完整、流畅的中文句子，像人在说话
3. 摘要里自然涵盖：异常类型、发生时间、影响范围、建议怎么处置
4. 处置建议要具体可执行，用「您可以先…再…」这类口语，不要写「请查看日志」"""

ALERT_SUMMARY_USER_TEMPLATE = """刚检测到一项系统异常，请用 JSON 生成一份用户能直接看懂的告警摘要：

{{
  "title": "一句话标题，口语化，不要英文代号",
  "summary": "2-3 句连贯叙述：刚才发生了什么、大概什么时候、会影响到哪些功能（把类型/时间/影响自然写进正文，不要列小标题）",
  "root_cause": "用「可能是…」「多半因为…」这类口吻解释原因",
  "suggestion": "告诉用户具体怎么做，用自然口语串联（可适度用 1.2.3.，但不要以「处理方法」开头）",
  "impact_scope": "一句话概括影响面，供系统归档用",
  "occurred_at": "发生时间描述（如：刚刚 / 过去几分钟内连续出现）"
}}

异常类型: {event_type}（含义：{event_type_cn}）
告警级别: {level}（info=提示, warning=警告, critical=严重）
当前时间: {now}
上下文数据: {context}
"""


ASSISTANT_SYSTEM = """你是车载视觉感知系统的告警助手「小智」，像一位熟悉系统的同事，帮用户理解异常并给出建议。

你能解读：车牌识别失败、手势识别不准、智能分析超时或额度不足、未授权访问、数据库异常、模型加载失败等。

回答风格：
- 先直接回应用户问题，再补充必要细节；语气亲切、自然，像微信里跟同事解释
- 结合上下文里的具体数字（失败次数、置信度、IP、额度等）个性化说明，不要套话
- 问原因就解释原因，问怎么办就给可执行建议，问影响就说对用户实际有什么影响
- 用中文大白话，2-4 段即可，不要写「处理方法：」「影响范围：」这类小标题
- 禁止 API 路径、Token、Webhook、SSE、unknown 等技术词；配置项用中文描述（如「群消息推送地址」）
- 纯文本输出，不要用 Markdown（禁止 **、#、* 等符号）"""


class LLMService:
    def _extract_tokens(self, resp_data: dict) -> int:
        usage = resp_data.get("usage") or {}
        return (
            usage.get("total_tokens")
            or (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
            or 0
        )

    async def _track_llm_response(self, resp_data: dict) -> None:
        tokens = self._extract_tokens(resp_data)
        from app.services.alert_agent import alert_agent

        db = SessionLocal()
        try:
            await alert_agent.track_llm_success(db, tokens_used=tokens)
        finally:
            db.close()

    async def _record_llm_failure(self, exc: Exception | None = None) -> None:
        from app.database import SessionLocal
        from app.services.alert_agent import alert_agent

        alert_agent.record_llm_call(success=False)
        db = SessionLocal()
        try:
            await alert_agent.check_and_alert(db, "llm")
        finally:
            db.close()

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """统一 OpenAI 兼容 Chat Completions 调用。"""
        if not settings.llm_configured:
            raise RuntimeError("LLM API Key 未配置")

        payload: dict[str, Any] = {
            "model": settings.effective_llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or settings.llm_max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
            resp = await client.post(
                f"{settings.effective_llm_base}/chat/completions",
                headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        await self._track_llm_response(data)
        return data

    async def test_connection(self) -> dict[str, Any]:
        """测试 LLM API 连通性（启动时或手动调用）。"""
        if not settings.llm_configured:
            return {
                "ok": False,
                "mode": "template",
                "message": "未配置 LLM_API_KEY，告警摘要将使用内置模板降级",
                "provider": settings.llm_provider,
            }

        try:
            data = await self.chat_completion(
                [
                    {"role": "system", "content": "你是系统助手，请用 JSON 回复。"},
                    {"role": "user", "content": '回复 JSON: {"status":"ok","message":"连接成功"}'},
                ],
                temperature=0,
                json_mode=True,
                max_tokens=64,
            )
            content = data["choices"][0]["message"]["content"]
            return {
                "ok": True,
                "mode": "llm",
                "message": "LLM API 连接正常",
                "provider": settings.llm_provider,
                "provider_label": settings.llm_provider_label,
                "model": settings.effective_llm_model,
                "base_url": settings.effective_llm_base,
                "sample_response": content[:200],
                "tokens_used": self._extract_tokens(data),
            }
        except Exception as e:
            await self._record_llm_failure(e)
            llm_logger.warning("LLM 连接测试失败: %s", e)
            return {
                "ok": False,
                "mode": "template",
                "message": f"LLM API 连接失败，将降级为模板告警: {e}",
                "provider": settings.llm_provider,
                "provider_label": settings.llm_provider_label,
                "model": settings.effective_llm_model,
                "base_url": settings.effective_llm_base,
                "error": str(e),
            }

    async def ask_assistant(self, question: str, context: dict[str, Any] | None = None) -> str:
        context = context or {}
        intent = detect_assistant_intent(question)
        knowledge = build_assistant_knowledge(context)

        if not settings.llm_configured:
            return self._template_assistant_answer(question, context, intent=intent)

        plan = knowledge["plan"]
        user_prompt = f"""用户问: {question}
（意图参考: {intent} — root_cause=问原因, action=问怎么办, severity=问紧不紧急, impact=问影响, status=问现状）

当前这条告警：
- 类型: {knowledge['event_name']}
- 级别: {knowledge.get('level', '提示')}
- 标题: {context.get('title', '')}
- 摘要: {context.get('summary', '')}
- 已有分析: {context.get('root_cause', '')}
- 已有建议: {context.get('suggestion', '')}
- 详情: {json.dumps(context.get('detail') or {}, ensure_ascii=False)}
- 补充: {knowledge.get('detail_hint', '')}

背景参考（请融入回答、结合详情改写，不要原样照搬）：
- 常见原因: {plan['root_cause']}
- 可参考做法: {_format_steps_conversational(knowledge['personalized_actions'])}
- 可能影响: {plan['impact']}

实时感知: {json.dumps(context.get('perception') or {}, ensure_ascii=False)}
系统概况: {json.dumps(context.get('system_status') or {}, ensure_ascii=False)}

请像同事聊天一样直接回答用户，不要列「处理方法」等小标题。"""

        try:
            data = await self.chat_completion(
                [
                    {"role": "system", "content": ASSISTANT_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.6,
            )
            answer = data["choices"][0]["message"]["content"]
            answer = humanize_tech_terms(self._strip_markdown(answer))
            if not answer or len(answer.strip()) < 8:
                return self._template_assistant_answer(question, context, intent=intent)
            if _is_useless_suggestion(answer):
                template = self._template_assistant_answer(question, context, intent=intent)
                if template and len(template) > len(answer):
                    return template
            return answer
        except Exception as e:
            await self._record_llm_failure(e)
            return self._template_assistant_answer(question, context, intent=intent)

    def _template_assistant_answer(
        self,
        question: str,
        context: dict[str, Any],
        *,
        intent: str | None = None,
    ) -> str:
        return assistant_answer_for_user(question, context, intent=intent)

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """移除 LLM 常输出的 Markdown 标记，避免前端显示原始 ** 符号。"""
        if not text:
            return text
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", text)
        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
        return text.strip()

    def _parse_json_response(self, content: str) -> dict[str, str] | None:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(content[start:end])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return None

    async def generate_alert_summary(
        self,
        event_type: str,
        level: str,
        context: dict[str, Any],
        *,
        force_template: bool = False,
    ) -> dict[str, str]:
        """通过 LLM API 生成结构化告警摘要；失败时降级为模板（不递归触发告警）。"""
        if force_template or not settings.llm_configured:
            return self._template_summary(event_type, level, context)

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        user_prompt = ALERT_SUMMARY_USER_TEMPLATE.format(
            event_type=event_type,
            event_type_cn=event_type_to_user(event_type),
            level=level,
            now=now,
            context=json.dumps(context, ensure_ascii=False),
        )

        try:
            data = await self.chat_completion(
                [
                    {"role": "system", "content": ALERT_SUMMARY_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.55,
                json_mode=True,
            )
            content = data["choices"][0]["message"]["content"]
            parsed = self._parse_json_response(content)
            if parsed and parsed.get("title") and parsed.get("summary"):
                return self._normalize_summary(parsed, event_type, level, context)
        except Exception as e:
            llm_logger.warning("LLM 告警摘要生成失败，降级模板: %s", e)
            await self._record_llm_failure(e)

        result = self._template_summary(event_type, level, context)
        result["_llm_failed"] = True
        return result

    def _normalize_summary(
        self,
        parsed: dict[str, Any],
        event_type: str,
        level: str,
        context: dict[str, Any],
    ) -> dict[str, str]:
        """合并 LLM 输出与模板兜底，确保字段完整。"""
        fallback = self._template_summary(event_type, level, context)
        summary = humanize_tech_terms(str(parsed.get("summary") or fallback["summary"]))
        impact = parsed.get("impact_scope")
        if impact and str(impact) not in summary:
            impact_text = humanize_tech_terms(str(impact))
            if summary.endswith(("。", "！", "？")):
                summary = f"{summary[:-1]}，可能会影响到{impact_text}。"
            else:
                summary = f"{summary}，可能会影响到{impact_text}。"

        return {
            "title": humanize_tech_terms(str(parsed.get("title") or fallback["title"])),
            "summary": summary,
            "root_cause": humanize_tech_terms(str(parsed.get("root_cause") or fallback["root_cause"])),
            "suggestion": humanize_tech_terms(str(parsed.get("suggestion") or fallback["suggestion"])),
        }

    def _template_summary(self, event_type: str, level: str, context: dict) -> dict[str, str]:
        plan = _get_plan(event_type)
        now_hint = context.get("timestamp") or "刚刚"
        templates = {
            "lpr_consecutive_failure": {
                "title": "车牌识别连续失败",
                "summary": (
                    f"系统检测到连续 {context.get('count', 5)} 次车牌识别失败（{now_hint}），"
                    f"道路感知中的车牌识别功能可能暂时不可用。"
                ),
                "root_cause": plan["root_cause"],
                "suggestion": _format_steps(plan["actions"]),
            },
            "lpr_high_failure_rate": {
                "title": "车牌识别失败率过高",
                "summary": (
                    f"最近 {context.get('window_seconds', 300)} 秒内车牌识别失败率约 "
                    f"{context.get('rate', '30%')}（{context.get('fails', '?')}/{context.get('total', '?')} 次），"
                    f"识别准确率明显下降。"
                ),
                "root_cause": plan["root_cause"],
                "suggestion": _format_steps(plan["actions"]),
            },
            "gesture_low_confidence": {
                "title": "手势识别置信度持续偏低",
                "summary": (
                    f"「{context.get('module', '手势')}」模块连续多次置信度低于 "
                    f"{context.get('threshold', 0.4):.0%}（当前约 {context.get('confidence', 0.3):.0%}），"
                    f"识别结果可能不可靠。"
                ),
                "root_cause": plan["root_cause"],
                "suggestion": _format_steps(plan["actions"]),
            },
            "llm_api_timeout": {
                "title": "智能分析响应较慢",
                "summary": "告警智能体调用大语言模型 API 超时或失败，已自动降级为模板告警。",
                "root_cause": plan["root_cause"],
                "suggestion": _format_steps(plan["actions"]),
            },
            "llm_token_exhausted": {
                "title": "智能分析额度即将耗尽",
                "summary": (
                    f"LLM Token 已使用 {context.get('used', '?')}/{context.get('limit', '?')} "
                    f"（{context.get('ratio', '80%')}），剩余 {context.get('remaining', '?')}。"
                ),
                "root_cause": "大语言模型 API 调用配额接近上限，继续调用可能失败。",
                "suggestion": "1. 检查 API 账户余额或配额\n2. 适当提高 alert_token_limit 配置\n3. 非紧急告警可暂时依赖模板模式",
            },
            "llm_token_exceeded": {
                "title": "智能分析额度已用完",
                "summary": f"LLM Token 配额已超额（{context.get('ratio', '100%')}），智能摘要功能已暂停。",
                "root_cause": "API 调用次数或 Token 用量达到账户上限。",
                "suggestion": "1. 充值或升级 API 套餐\n2. 系统将继续使用模板告警，不影响基础监控",
            },
            "unauthorized_access": {
                "title": "未授权访问尝试",
                "summary": (
                    f"检测到来自 {context.get('ip', '未知')} 的未授权 API 访问"
                    f"（路径: {context.get('path', '未知')}），"
                    f"近 {context.get('window_seconds', 300)} 秒内累计 {context.get('count', 1)} 次。"
                ),
                "root_cause": plan["root_cause"],
                "suggestion": _format_steps(plan["actions"]),
            },
            "service_unhealthy": {
                "title": "系统服务健康异常",
                "summary": f"服务「{context.get('service', '未知')}」状态异常：{context.get('detail', '需检查')}。",
                "root_cause": "系统组件运行异常，可能影响相关功能。",
                "suggestion": "1. 查看告警中心了解详情\n2. 重启相关服务\n3. 检查数据库连接",
            },
            "model_load_failure": {
                "title": "AI 模型加载失败",
                "summary": f"模型「{context.get('model_name', '未知')}」加载失败，相关识别功能不可用。",
                "root_cause": f"错误类型: {context.get('error_type', '未知')}，详情: {context.get('error', '未知')}",
                "suggestion": "1. 确认模型文件已下载到 models 目录\n2. 检查磁盘空间与文件权限\n3. 重启服务后重试",
            },
            "database_connection_error": {
                "title": "数据库连接异常",
                "summary": f"连续 {context.get('consecutive_fails', 3)} 次数据库连接失败，数据读写可能中断。",
                "root_cause": "数据库服务未启动、连接字符串错误或网络异常。",
                "suggestion": "1. 检查 SQL Server / SQLite 文件是否存在\n2. 验证 DATABASE_URL 配置\n3. 重启数据库服务",
            },
            "webhook_delivery_failure": {
                "title": "群消息推送失败",
                "summary": (
                    f"刚才往群里发告警消息时失败了（近 {context.get('window', '几次')} "
                    f"约 {context.get('fails', '?')} 次没发出去），群里可能暂时收不到推送。"
                ),
                "root_cause": plan["root_cause"],
                "suggestion": _format_steps_conversational(plan["actions"]),
            },
            "email_delivery_failure": {
                "title": "邮件通知发送失败",
                "summary": (
                    f"邮件通知最近发送不太顺利（近 {context.get('window', '几次')} "
                    f"约 {context.get('fails', '?')} 次失败），邮箱可能收不到提醒。"
                ),
                "root_cause": plan["root_cause"],
                "suggestion": _format_steps_conversational(plan["actions"]),
            },
            "config_missing": {
                "title": "系统配置不完整",
                "summary": (
                    f"系统发现「{humanize_tech_terms(str(context.get('config_key', '某项配置')))}」还没设置好，"
                    f"部分功能可能会受影响。"
                ),
                "root_cause": plan["root_cause"],
                "suggestion": _format_steps_conversational(plan["actions"]),
            },
            "test_event": {
                "title": "这是一条测试提醒",
                "summary": plan["root_cause"],
                "root_cause": "这是功能演示，不代表真实故障。",
                "suggestion": _format_steps(plan["actions"]),
            },
        }
        base = templates.get(event_type, {
            "title": event_type_to_user(event_type),
            "summary": f"刚才检测到{event_type_to_user(event_type)}（{level_to_user(level)}级别），建议留意一下。",
            "root_cause": plan["root_cause"],
            "suggestion": _format_steps_conversational(plan["actions"]),
        })
        return {k: humanize_tech_terms(v) if isinstance(v, str) else v for k, v in base.items()}

    def get_provider_options(self) -> list[dict[str, str]]:
        """返回支持的 LLM 厂商列表（供前端/文档展示）。"""
        return [
            {"key": key, "label": val["label"], "base": val["base"], "model": val["model"]}
            for key, val in LLM_PROVIDER_PRESETS.items()
        ]


llm_service = LLMService()

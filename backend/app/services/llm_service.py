import json
import re
from typing import Any
import httpx

from app.config import settings, LLM_PROVIDER_PRESETS
from app.database import SessionLocal
from app.utils.logger import get_logger, local_now_cn
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
4. 时间描述优先用「刚刚」「过去几分钟内」等相对说法，不要写具体钟点（界面会单独显示精确时间）
5. 处置建议要具体可执行，用「您可以先…再…」这类口语，不要写「请查看日志」"""

ALERT_SUMMARY_USER_TEMPLATE = """刚检测到一项系统异常，请用 JSON 生成一份用户能直接看懂的告警摘要：

{{
  "title": "一句话标题，口语化，不要英文代号",
  "summary": "2-3 句连贯叙述：刚才发生了什么、大概什么时候（用「刚刚/几分钟前」，勿写「八点三十分」这类钟点）、会影响到哪些功能（把类型/时间/影响自然写进正文，不要列小标题）",
  "root_cause": "用「可能是…」「多半因为…」这类口吻解释原因",
  "suggestion": "告诉用户具体怎么做，用自然口语串联（可适度用 1.2.3.，但不要以「处理方法」开头）",
  "impact_scope": "一句话概括影响面，供系统归档用",
  "occurred_at": "发生时间描述（如：刚刚 / 过去几分钟内连续出现，勿写具体钟点）"
}}

异常类型: {event_type}（含义：{event_type_cn}）
告警级别: {level}（info=提示, warning=警告, critical=严重）
当前时间（北京时间，仅供理解时效，摘要里勿写具体钟点）: {now}
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
    async def generate_alert_summary(
        self,
        event_type: str,
        level: str,
        context: dict[str, Any],
    ) -> dict[str, str]:
        if not settings.llm_api_key:
            return self._template_summary(event_type, level, context)

        now = local_now_cn()
        user_prompt = ALERT_SUMMARY_USER_TEMPLATE.format(
            event_type=event_type,
            event_type_cn=event_type_to_user(event_type),
            level=level,
            now=now,
            context=json.dumps(context, ensure_ascii=False),
        )

异常类型: {event_type}
告警级别: {level}
上下文: {json.dumps(context, ensure_ascii=False)}
"""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{settings.llm_api_base.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                    json={
                        "model": settings.llm_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                    },
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    return json.loads(content[start:end])
        except Exception as e:
            llm_logger.warning("LLM 告警摘要生成失败，降级模板: %s", e)
            await self._record_llm_failure(e)

        result = self._template_summary(event_type, level, context)
        result["_llm_failed"] = True
        return result

    @staticmethod
    def _merge_impact_scope(summary: str, impact: Any) -> str:
        """将 impact_scope 并入摘要，避免与 LLM 已有表述重复或「影响到影响…」。"""
        if not impact:
            return summary
        impact_text = humanize_tech_terms(str(impact)).strip()
        if not impact_text:
            return summary
        if impact_text in summary:
            return summary

        core = impact_text
        for prefix in ("影响到", "影响"):
            if core.startswith(prefix):
                core = core[len(prefix):].strip()
                break
        if core and core in summary:
            return summary
        if "影响" in summary and impact_text.startswith("影响"):
            return summary
        if re.search(r"可能(会)?影响", summary):
            return summary

        fragment = core or impact_text
        if summary.endswith(("。", "！", "？")):
            return f"{summary[:-1]}，可能会影响{fragment}。"
        return f"{summary}，可能会影响{fragment}。"

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
        summary = self._merge_impact_scope(summary, parsed.get("impact_scope"))

        return {
            "title": humanize_tech_terms(str(parsed.get("title") or fallback["title"])),
            "summary": summary,
            "root_cause": humanize_tech_terms(str(parsed.get("root_cause") or fallback["root_cause"])),
            "suggestion": humanize_tech_terms(str(parsed.get("suggestion") or fallback["suggestion"])),
        }

    def _template_summary(self, event_type: str, level: str, context: dict) -> dict[str, str]:
        templates = {
            "lpr_consecutive_failure": {
                "title": "车牌识别连续失败",
                "summary": f"系统检测到连续 {context.get('count', 5)} 次车牌识别失败，可能影响道路感知功能。",
                "root_cause": "可能原因：摄像头遮挡、光照不足、模型加载异常或输入图像质量过低。",
                "suggestion": "检查摄像头状态，确认模型服务正常，尝试更换输入源或调整曝光参数。",
            },
            "gesture_low_confidence": {
                "title": "手势识别置信度持续偏低",
                "summary": f"手势识别模块置信度低于阈值 ({context.get('confidence', 0.3):.0%})，识别结果可能不可靠。",
                "root_cause": "可能原因：手部/人体未完整入镜、背景干扰、光照变化或遮挡。",
                "suggestion": "调整摄像头角度，改善光照条件，确保目标完整可见。",
            },
            "llm_api_timeout": {
                "title": "LLM API 调用超时",
                "summary": "告警智能体调用大语言模型 API 超时，自动降级为模板告警。",
                "root_cause": "网络延迟、API 服务不可用或 Token 配额不足。",
                "suggestion": "检查 API 密钥与网络连接，确认配额余额，必要时切换备用模型。",
            },
            "unauthorized_access": {
                "title": "未授权访问尝试",
                "summary": f"检测到来自 {context.get('ip', '未知')} 的未授权 API 访问。",
                "root_cause": "无效或过期的访问令牌，或恶意扫描行为。",
                "suggestion": "审查访问日志，更新密钥策略，必要时封禁 IP。",
            },
        }
        base = templates.get(event_type, {
            "title": f"系统异常: {event_type}",
            "summary": f"检测到 {event_type} 事件，级别: {level}",
            "root_cause": "待进一步分析",
            "suggestion": "请查看系统日志获取详细信息",
        })
        return base


llm_service = LLMService()

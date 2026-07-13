"""三路感知融合驾驶建议 —— 模板兜底 + LLM 增强。"""

from __future__ import annotations

from typing import Any

from app.utils.scenario_rules import (
    OWNER_ACTION_CN,
    POLICE_GESTURE_CN,
    normalize_plate_labels,
)


def _signal_parts(correlated: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    police = correlated.get("police_gesture")
    police_cn = correlated.get("police_gesture_cn") or POLICE_GESTURE_CN.get(police or "", "")
    if police and police != "no_gesture" and police_cn:
        parts.append(f"交警{police_cn}")

    plates = normalize_plate_labels(correlated.get("plates"))
    if plates:
        parts.append(f"检测到车牌{'/'.join(plates[:2])}")

    owner_cn = correlated.get("owner_action_cn") or OWNER_ACTION_CN.get(
        correlated.get("owner_action") or "", ""
    )
    if owner_cn:
        parts.append(f"车主{owner_cn}")

    owner_gesture_cn = correlated.get("owner_gesture_cn")
    if owner_gesture_cn and not owner_cn:
        parts.append(f"车主手势{owner_gesture_cn}")

    return parts


def build_template_driving_advice(
    correlated: dict[str, Any],
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """基于三路感知生成综合驾驶建议（无 LLM 时的规则模板）。"""
    police = correlated.get("police_gesture")
    police_cn = correlated.get("police_gesture_cn") or POLICE_GESTURE_CN.get(police or "", "")
    plates = normalize_plate_labels(correlated.get("plates"))
    plate_text = "/".join(plates[:2]) if plates else ""
    owner_action = correlated.get("owner_action")
    owner_cn = correlated.get("owner_action_cn") or OWNER_ACTION_CN.get(owner_action or "", "")

    signals = _signal_parts(correlated)
    signals_summary = " + ".join(signals) if signals else "暂无明显驾驶信号"

    if not signals:
        return {
            "advice": "当前暂无明显驾驶相关信号，请保持正常行驶并持续关注路况。",
            "signals_summary": signals_summary,
            "priority": "normal",
            "mode": "template",
        }

    advice = ""
    priority = "normal"

    if police == "stop":
        priority = "high"
        if plate_text:
            advice = (
                f"前方交警停止手势，且检测到前车车牌{plate_text}，"
                f"建议减速并保持安全车距，准备停车避让。"
            )
        else:
            advice = "前方交警示意停止，建议立即减速并准备停车，双手保持方向盘控制。"
        if owner_action:
            advice += f" 同时暂停车内「{owner_cn}」操作，优先服从道路指挥。"

    elif police == "pull_over":
        priority = "high"
        advice = "交警示意靠边停车，请打右转向灯，观察后视镜后缓慢靠边。"
        if plate_text:
            advice += f" 注意与前车{plate_text}保持安全距离。"
        if owner_action == "go_home":
            advice += " 车机返回待机可待停稳后再操作。"

    elif police == "slow_down":
        priority = "medium"
        advice = "交警示意减速慢行，请降低车速，避免急刹。"
        if plate_text:
            advice += f" 前方车辆{plate_text}，注意跟车距离。"
        if owner_action == "volume_adjust":
            advice += " 建议暂停音量调节，避免分心。"

    elif police in ("turn_left", "turn_right", "lane_change"):
        priority = "medium"
        turn_cn = police_cn or "转向"
        advice = f"交警示意{turn_cn}，请提前打转向灯并观察侧后方来车。"
        if plate_text:
            advice += f" 结合前车{plate_text}位置判断变道空间。"
        if owner_action in ("prev_page", "next_page"):
            advice += " 转向过程中请暂停翻页操作。"

    elif police == "go_straight":
        advice = "交警示意直行，请保持车道匀速通过路口。"
        if plate_text:
            advice += f" 注意跟随前车{plate_text}，保持安全车距。"
        if owner_action in ("answer_call", "hang_up"):
            advice += " 通话操作请尽量简短，勿分散注意力。"

    elif police == "left_turn_wait":
        advice = "交警示意左转弯待转，请在待转区等候，勿抢行。"
        if plate_text:
            advice += f" 关注前车{plate_text}动向，待明确放行后再起步。"

    elif plate_text and not police:
        advice = f"检测到前方车辆车牌{plate_text}，请保持安全车距，随时准备制动。"

    elif owner_action and not police:
        advice = f"车内正在{owner_cn}，请确保不影响驾驶注意力；复杂路况下建议暂停操作。"
        if plate_text:
            advice += f" 同时注意前车{plate_text}。"

    else:
        advice = f"综合当前信号（{signals_summary}），请谨慎驾驶，服从道路指挥优先于车内操作。"

    return {
        "advice": advice,
        "signals_summary": signals_summary,
        "priority": priority,
        "mode": "template",
        "sources": {
            "police_gesture": police,
            "police_gesture_cn": police_cn,
            "plates": plates,
            "owner_action": owner_action,
            "owner_action_cn": owner_cn,
        },
    }

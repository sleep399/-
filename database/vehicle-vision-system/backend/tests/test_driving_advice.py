"""三路感知融合驾驶建议与被动日志聚合测试。"""

import asyncio
from datetime import timedelta

from app.config import settings
from app.services.llm_service import LLMService
from app.services.scenario_fusion_service import ScenarioFusionService
from app.utils.driving_advice import build_template_driving_advice


def test_stop_with_plate_generates_yield_advice():
    result = build_template_driving_advice({
        "police_gesture": "stop",
        "police_gesture_cn": "停止",
        "plates": [{"plate_number": "京A12345"}],
    })
    assert "停止" in result["advice"]
    assert "京A12345" in result["advice"]
    assert "减速" in result["advice"]
    assert result["priority"] == "high"


def test_empty_signals_fallback():
    result = build_template_driving_advice({})
    assert "暂无明显" in result["advice"]


def test_three_signal_sources_are_summarized_together():
    result = build_template_driving_advice({
        "police_gesture": "slow_down",
        "police_gesture_cn": "减速慢行",
        "plates": ["粤B12345"],
        "owner_action": "volume_adjust",
    })
    assert "交警减速慢行" in result["signals_summary"]
    assert "检测到车牌粤B12345" in result["signals_summary"]
    assert "车主调节音量/温度" in result["signals_summary"]
    assert "暂停音量调节" in result["advice"]
    assert result["priority"] == "medium"


def test_passive_log_ingestion_collects_signals_without_conflict_side_effects():
    service = ScenarioFusionService()

    async def collect_signals():
        assert await service.ingest_lpr(
            None,
            success=True,
            plate_count=1,
            plates=["粤B12345"],
            source="test",
            evaluate_conflicts=False,
        ) is None
        assert await service.ingest_police(
            None,
            gesture="slow_down",
            gesture_cn="减速慢行",
            confidence=0.91,
            source="test",
            evaluate_conflicts=False,
        ) is None
        assert await service.ingest_owner(
            None,
            gesture="circle",
            gesture_cn="单指画圈",
            action="volume_adjust",
            confidence=0.88,
            source="test",
            evaluate_conflicts=False,
        ) is None

    asyncio.run(collect_signals())

    snapshot = service.get_snapshot()
    assert snapshot["lpr"]["plates"] == ["粤B12345"]
    assert snapshot["police"]["gesture"] == "slow_down"
    assert snapshot["owner"]["action"] == "volume_adjust"
    assert snapshot["open_conflicts"] == 0
    assert snapshot["owner_suppressed"] is False


def test_failed_lpr_and_low_confidence_police_are_not_used_for_advice():
    service = ScenarioFusionService()

    async def collect_invalid_signals():
        await service.ingest_lpr(
            None,
            success=False,
            plate_count=0,
            plates=[{"plate_number": "未识别"}],
            evaluate_conflicts=False,
        )
        await service.ingest_police(
            None,
            gesture="stop",
            gesture_cn="停止",
            confidence=0.2,
            evaluate_conflicts=False,
        )

    asyncio.run(collect_invalid_signals())
    correlated = service._build_correlation_snapshot()
    assert correlated["plates"] == []
    assert correlated["police_gesture"] is None


def test_snapshot_drops_signals_outside_the_configured_window():
    service = ScenarioFusionService()
    service._record_event("lpr", {
        "success": True,
        "plate_count": 1,
        "plates": ["京A12345"],
        "updated_at": service._now().isoformat(),
    })
    service._events[0]["timestamp"] = service._now() - timedelta(
        seconds=settings.scenario_window_seconds + 1
    )

    snapshot = service.get_snapshot()
    assert snapshot["lpr"]["plate_count"] == 0
    assert snapshot["lpr"]["plates"] == []


def test_llm_priority_is_restricted_to_frontend_safe_values():
    service = LLMService()
    original_api_key = settings.llm_api_key

    async def fake_chat_completion(*args, **kwargs):
        return {
            "choices": [{
                "message": {
                    "content": '{"advice":"保持车距","signals_summary":"前车",'
                               '"priority":"evil"}'
                }
            }]
        }

    service.chat_completion = fake_chat_completion
    settings.llm_api_key = "test-only"
    try:
        result = asyncio.run(service.generate_driving_advice({"plates": ["粤B12345"]}))
    finally:
        settings.llm_api_key = original_api_key

    assert result["mode"] == "llm"
    assert result["priority"] == "normal"

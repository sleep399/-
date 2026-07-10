import asyncio
import base64
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.database import SessionLocal
from app.services.lpr_service import lpr_service
from app.services.police_gesture_service import police_gesture_service
from app.services.owner_gesture_service import owner_gesture_service
from app.services.alert_agent import alert_agent
from app.utils.logger import write_log, log_exception

router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    await websocket.accept()
    alert_agent.register_ws(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        alert_agent.unregister_ws(websocket)


async def _process_stream_frame(module: str, service, img_bytes: bytes) -> dict:
    """处理实时流识别帧，写入日志并触发告警检测"""
    db = SessionLocal()
    try:
        if module == "lpr":
            try:
                result = service.recognize(img_bytes)
            except FileNotFoundError as e:
                await alert_agent.handle_model_load_failure(db, "fh02.pth", e)
                raise
            success = result.get("success", False) and result.get("plate_count", 0) > 0
            alert_agent.record_lpr_result(success)
            await alert_agent.check_and_alert(db, "lpr")
            level = "INFO" if success else "WARN"
            write_log(
                db, "lpr",
                f"[WebSocket流] 识别到 {result.get('plate_count', 0)} 个车牌",
                level=level,
                detail={"source": "websocket", "plates": result.get("plates", []), "success": success},
            )
            return result

        try:
            result = service.recognize(img_bytes)
        except FileNotFoundError as e:
            model_name = "hand_landmarker.task" if module == "owner" else "pose_landmarker_lite.task"
            await alert_agent.handle_model_load_failure(db, model_name, e)
            raise

        category = "owner_gesture" if module == "owner" else "police_gesture"
        alert_agent.record_gesture_confidence(module, result["confidence"])
        await alert_agent.check_and_alert(db, module)
        level = "INFO" if result["confidence"] >= 0.4 else "WARN"
        write_log(
            db, category,
            f"[WebSocket流] 识别手势: {result['gesture_cn']} ({result['confidence']:.0%})",
            level=level,
            detail={"source": "websocket", "gesture": result["gesture"], "confidence": result["confidence"]},
        )
        return result
    except FileNotFoundError:
        raise
    except Exception as e:
        category = "lpr" if module == "lpr" else ("owner_gesture" if module == "owner" else "police_gesture")
        log_exception(db, category, f"[WebSocket流] {module} 识别失败", e)
        raise
    finally:
        db.close()


@router.websocket("/ws/stream/{module}")
async def ws_stream(websocket: WebSocket, module: str):
    """实时视频流识别: module = lpr | police | owner"""
    await websocket.accept()
    services = {"lpr": lpr_service, "police": police_gesture_service, "owner": owner_gesture_service}
    if module not in services:
        await websocket.send_json({"error": "无效模块"})
        await websocket.close()
        return

    service = services[module]
    db = SessionLocal()
    try:
        write_log(db, "system", f"WebSocket 实时流连接: {module}", detail={"module": module})
    finally:
        db.close()

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "frame":
                img_bytes = base64.b64decode(msg["data"])
                result = await _process_stream_frame(module, service, img_bytes)
                await websocket.send_json({"type": "result", "module": module, "data": result})
            elif msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        db = SessionLocal()
        try:
            write_log(db, "system", f"WebSocket 实时流断开: {module}", detail={"module": module})
        finally:
            db.close()
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass

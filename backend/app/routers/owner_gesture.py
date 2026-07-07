import json
import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.records import OwnerGestureRecord, VehicleState
from app.schemas import GestureResponse, VehicleStateResponse
from app.services.owner_gesture_service import owner_gesture_service, OWNER_GESTURES
from app.services.alert_agent import alert_agent
from app.utils.auth import get_current_user
from app.utils.logger import write_log
from app.config import settings

router = APIRouter(prefix="/api/owner-gesture", tags=["车主手势控车"])


def _get_or_create_state(db: Session, user_id: int | None) -> VehicleState:
    state = db.query(VehicleState).filter(VehicleState.user_id == user_id).first()
    if not state:
        state = VehicleState(user_id=user_id)
        db.add(state)
        db.commit()
        db.refresh(state)
    return state


@router.post("/recognize", response_model=GestureResponse, summary="识别车主手势并触发控车")
async def recognize(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    content = await file.read()
    try:
        result = owner_gesture_service.recognize(content)
    except Exception as e:
        write_log(db, "owner_gesture", f"识别失败: {e}", level="ERROR")
        raise HTTPException(500, str(e))

    alert_agent.record_gesture_confidence("owner", result["confidence"])
    await alert_agent.check_and_alert(db, "owner")

    user_id = user.id if user else None
    if result.get("action"):
        state = _get_or_create_state(db, user_id)
        state_dict = {
            "volume": state.volume,
            "temperature": state.temperature,
            "phone_status": state.phone_status,
            "current_page": state.current_page,
            "is_awake": state.is_awake,
        }
        updated = owner_gesture_service.apply_action_to_state(result["action"], state_dict)
        state.volume = updated["volume"]
        state.temperature = updated["temperature"]
        state.phone_status = updated["phone_status"]
        state.current_page = updated["current_page"]
        state.is_awake = updated["is_awake"]
        state.updated_at = datetime.utcnow()
        db.commit()
        write_log(db, "owner_gesture", f"手势触发: {result['gesture_cn']} -> {result['action']}")

    save_path = settings.upload_dir / "owner" / f"{uuid.uuid4().hex}.jpg"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(content)

    record = OwnerGestureRecord(
        user_id=user_id,
        source_type="image",
        image_path=str(save_path),
        gesture=result["gesture"],
        gesture_cn=result["gesture_cn"],
        confidence=result["confidence"],
        action=result.get("action"),
        keypoints_json=json.dumps(result["keypoints"], ensure_ascii=False),
        annotated_image=result["annotated_image"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return GestureResponse(**result, record_id=record.id)


@router.get("/vehicle-state", response_model=VehicleStateResponse, summary="获取模拟车辆状态")
def get_vehicle_state(db: Session = Depends(get_db), user=Depends(get_current_user)):
    state = _get_or_create_state(db, user.id if user else None)
    return VehicleStateResponse(
        volume=state.volume,
        temperature=state.temperature,
        phone_status=state.phone_status,
        current_page=state.current_page,
        is_awake=state.is_awake,
    )


@router.put("/vehicle-state", response_model=VehicleStateResponse, summary="手动更新车辆状态")
def update_vehicle_state(
    data: VehicleStateResponse,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    state = _get_or_create_state(db, user.id if user else None)
    state.volume = data.volume
    state.temperature = data.temperature
    state.phone_status = data.phone_status
    state.current_page = data.current_page
    state.is_awake = data.is_awake
    state.updated_at = datetime.utcnow()
    db.commit()
    write_log(db, "owner_gesture", "手动更新车辆状态", detail=data.model_dump())
    return data


@router.get("/gestures", summary="支持的手势列表")
def gesture_list():
    seen = set()
    items = []
    for key, (en, cn, action) in OWNER_GESTURES.items():
        if key == "no_gesture" or key in seen:
            continue
        seen.add(key)
        items.append({"key": key, "en": en, "cn": cn, "action": action})
    return items


@router.get("/history", summary="历史记录")
def history(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    records = db.query(OwnerGestureRecord).order_by(OwnerGestureRecord.created_at.desc()).offset(skip).limit(limit).all()
    return [
        {
            "id": r.id,
            "gesture": r.gesture,
            "gesture_cn": r.gesture_cn,
            "confidence": r.confidence,
            "action": r.action,
            "annotated_image": r.annotated_image,
            "created_at": r.created_at.isoformat(),
        }
        for r in records
    ]
# 原有代码全部保留，只替换下面ws部分
from fastapi import WebSocket, WebSocketDisconnect
import numpy as np
import cv2
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 线程池，把同步mediapipe计算丢到后台线程，不阻塞主事件循环
executor = ThreadPoolExecutor(max_workers=2)

@router.websocket("/ws-stream")
async def gesture_websocket(websocket: WebSocket, db: Session = Depends(get_db), user=Depends(get_current_user)):
    await websocket.accept()
    user_id = user.id
    try:
        while True:
            frame_bytes = await websocket.receive_bytes()
            arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            # 关键修复：同步识别放到线程池执行，释放async事件循环
            result = await asyncio.get_event_loop().run_in_executor(
                executor,
                owner_gesture_service.recognize_frame,
                frame
            )
            if result.get("action"):
                state = _get_or_create_state(db, user_id)
                state_dict = {
                    "volume": state.volume,
                    "temperature": state.temperature,
                    "phone_status": state.phone_status,
                    "current_page": state.current_page,
                    "is_awake": state.is_awake,
                }
                updated = owner_gesture_service.apply_action_to_state(result["action"], state_dict)
                state.volume = updated["volume"]
                state.temperature = updated["temperature"]
                state.phone_status = updated["phone_status"]
                state.current_page = updated["current_page"]
                state.is_awake = updated["is_awake"]
                state.updated_at = datetime.utcnow()
                db.commit()
            resp = GestureResponse(**result)
            await websocket.send_json(resp.model_dump())
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print("ws error:", e)
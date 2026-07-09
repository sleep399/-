from __future__ import annotations
import json
import uuid
from pathlib import Path
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.records import LicensePlateRecord
from app.schemas import LPRResponse
from app.services.lpr_service import lpr_service
from app.services.alert_agent import alert_agent
from app.utils.auth import get_current_user
from app.utils.crypto import encrypt_json, decrypt_json
from app.utils.logger import write_log
from app.utils.video import process_video_file
from app.config import settings

router = APIRouter(prefix="/api/lpr", tags=["车牌识别"])


@router.post("/recognize", response_model=LPRResponse, summary="上传图片识别车牌")
async def recognize_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    content = await file.read()
    filename = file.filename or ""
    try:
        result = lpr_service.recognize(content, filename)
    except Exception as e:
        alert_agent.record_lpr_result(False)
        await alert_agent.check_and_alert(db, "lpr")
        write_log(db, "lpr", f"识别失败: {e}", level="ERROR", user_id=user.id if user else None)
        raise HTTPException(500, str(e))

    alert_agent.record_lpr_result(result["success"])
    await alert_agent.check_and_alert(db, "lpr")

    save_name = f"{uuid.uuid4().hex}.jpg"
    save_path = settings.upload_dir / "lpr" / save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(content)

    encrypted_plates = encrypt_json({"plates": result["plates"]})
    record = LicensePlateRecord(
        user_id=user.id if user else None,
        source_type="image",
        image_path=str(save_path),
        annotated_image=result["annotated_image"],
        plates_json=encrypted_plates,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    write_log(db, "lpr", f"识别�?{result['plate_count']} 个车�?, detail={"plates": result["plates"]}, user_id=user.id if user else None)
    return LPRResponse(**result, record_id=record.id)


@router.post("/recognize-video", summary="上传视频识别车牌")
async def recognize_video(
    file: UploadFile = File(...),
    interval: int = Query(15, ge=1, le=120),
    max_results: int = Query(60, ge=1, le=300),
    max_sampled_frames: int = Query(120, ge=1, le=1000),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    content = await file.read()
    save_path = settings.upload_dir / "lpr" / f"{uuid.uuid4().hex}.mp4"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(content)

    result = process_video_file(lpr_service, save_path, interval, max_results, max_sampled_frames)
    write_log(db, "lpr", f"视频识别完成，有效帧 {result['result_count']}", user_id=user.id if user else None)
    return result


@router.get("/history", summary="历史识别记录")
def history(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    q = db.query(LicensePlateRecord).order_by(LicensePlateRecord.created_at.desc())
    if user:
        q = q.filter((LicensePlateRecord.user_id == user.id) | (LicensePlateRecord.user_id.is_(None)))
    records = q.offset(skip).limit(limit).all()
    return [
        {
            "id": r.id,
            "source_type": r.source_type,
            "plate_count": len(decrypt_json(r.plates_json).get("plates", [])) if r.plates_json else 0,
            "annotated_image": r.annotated_image,
            "created_at": r.created_at.isoformat(),
        }
        for r in records
    ]


@router.get("/ccpd-sample", summary="�?CCPD 数据集获取样本图片路�?)
def ccpd_sample(db: Session = Depends(get_db)):
    ccpd_path = (settings.base_dir / settings.ccpd_data_path).resolve()
    split_file = ccpd_path / "split" / "test.txt"
    if not split_file.exists():
        return {"samples": [], "message": "CCPD split 文件存在，请将图片数据放置于 CCPD 目录下对应子文件�?}
    lines = split_file.read_text(encoding="utf-8").strip().split("\n")[:10]
    samples = []
    for line in lines:
        img_path = ccpd_path / line.strip()
        samples.append({"relative": line.strip(), "exists": img_path.exists(), "full_path": str(img_path)})
    return {"samples": samples, "ccpd_root": str(ccpd_path)}

from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    username: str
    password: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None


class UserLogin(BaseModel):
    username: str
    password: str


class CodeLoginRequest(BaseModel):
    target: str
    code: str
    target_type: str = "email"


class SendCodeRequest(BaseModel):
    target: str
    target_type: str = "email"


class GestureResponse(BaseModel):
    gesture: str
    gesture_cn: str
    confidence: float
    annotated_image: str
    keypoints: list
    success: bool
    record_id: Optional[int] = None
    action: Optional[str] = None


class AlertResponse(BaseModel):
    id: int
    level: str
    event_type: str
    title: str
    summary: str
    root_cause: Optional[str] = None
    suggestion: Optional[str] = None
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class LogResponse(BaseModel):
    id: int
    category: str
    level: str
    message: str
    created_at: datetime

    class Config:
        from_attributes = True

from __future__ import annotations
import random
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User, VerificationCode
from app.schemas import Token, UserCreate, UserLogin, CodeLoginRequest, SendCodeRequest
from app.utils.auth import hash_password, verify_password, create_access_token, require_user

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=Token)
def register(data: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == data.username).first():
        raise HTTPException(400, "username already exists")
    user = User(username=data.username, email=data.email, phone=data.phone, hashed_password=hash_password(data.password), is_active=True)
    db.add(user)
    db.commit()
    return Token(access_token=create_access_token({"sub": user.username}))


@router.post("/login", response_model=Token)
def login(data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == data.username).first()
    if not user or not verify_password(data.password, user.hashed_password or ""):
        raise HTTPException(401, "invalid username or password")
    return Token(access_token=create_access_token({"sub": user.username}))


@router.post("/send-code")
def send_code(data: SendCodeRequest, db: Session = Depends(get_db)):
    code = f"{random.randint(100000, 999999)}"
    vc = VerificationCode(target=data.target, code=code, purpose="login", expires_at=datetime.utcnow() + timedelta(minutes=5))
    db.add(vc)
    db.commit()
    return {"message": "code sent", "code": code, "expires_in": 300}


@router.post("/login-code", response_model=Token)
def login_with_code(data: CodeLoginRequest, db: Session = Depends(get_db)):
    vc = db.query(VerificationCode).filter(
        VerificationCode.target == data.target,
        VerificationCode.used == False,
        VerificationCode.expires_at > datetime.utcnow(),
    ).order_by(VerificationCode.id.desc()).first()
    if not vc or vc.code != data.code:
        raise HTTPException(400, "invalid or expired code")
    vc.used = True
    field = User.email if data.target_type == "email" else User.phone
    user = db.query(User).filter(field == data.target).first()
    if not user:
        username = data.target.split("@")[0] if "@" in data.target else data.target
        user = User(username=username, email=data.target if data.target_type == "email" else None, phone=data.target if data.target_type == "phone" else None, is_active=True)
        db.add(user)
        db.flush()
    db.commit()
    return Token(access_token=create_access_token({"sub": user.username}))


@router.get("/me")
def me(user: User = Depends(require_user)):
    return {"id": user.id, "username": user.username, "email": user.email, "phone": user.phone}

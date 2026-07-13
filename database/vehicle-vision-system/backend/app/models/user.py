from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, Index, Integer, String, text
from app.database import Base


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        Index(
            "ix_users_phone",
            "phone",
            unique=True,
            mssql_where=text("phone IS NOT NULL"),
            sqlite_where=text("phone IS NOT NULL"),
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    email = Column(String(128), unique=True, index=True, nullable=True)
    phone = Column(String(20), nullable=True)
    hashed_password = Column(String(256), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class VerificationCode(Base):
    __tablename__ = "verification_codes"

    id = Column(Integer, primary_key=True, index=True)
    target = Column(String(128), index=True, nullable=False)
    code = Column(String(8), nullable=False)
    purpose = Column(String(32), default="login")
    expires_at = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False)


class WechatLoginSession(Base):
    __tablename__ = "wechat_login_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), unique=True, index=True)
    status = Column(String(16), default="pending")
    user_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

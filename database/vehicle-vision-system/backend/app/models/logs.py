from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text
from app.database import Base


class SystemLog(Base):
    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, index=True)
    category = Column(String(32), index=True)
    level = Column(String(16), default="INFO")
    message = Column(Text, nullable=False)
    detail_json = Column(Text, nullable=True)
    user_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

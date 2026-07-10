from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from app.config import settings

settings.data_dir.mkdir(parents=True, exist_ok=True)

db_url = settings.db_url
connect_args = {}
if db_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
elif db_url.startswith("mssql"):
    connect_args = {"autocommit": False}

engine = create_engine(
    db_url,
    echo=settings.database_echo,
    future=True,
    connect_args=connect_args,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    from app.services.alert_agent import alert_agent

    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
        alert_agent.record_db_connection(True)
        yield db
    except Exception:
        alert_agent.record_db_connection(False)
        raise
    finally:
        db.close()


def _migrate_schema():
    """为已有数据库补充新增列（SQLite 兼容）"""
    insp = inspect(engine)
    if "alert_events" not in insp.get_table_names():
        return
    existing = {c["name"] for c in insp.get_columns("alert_events")}
    with engine.begin() as conn:
        if "system_health_json" not in existing:
            conn.execute(text("ALTER TABLE alert_events ADD COLUMN system_health_json TEXT"))
        if "resolution_note" not in existing:
            conn.execute(text("ALTER TABLE alert_events ADD COLUMN resolution_note TEXT"))


def check_db_connection() -> bool:
    """检测数据库连接是否可用"""
    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
    finally:
        db.close()


def init_db():
    from app.models import user, records, logs, alerts  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _migrate_schema()

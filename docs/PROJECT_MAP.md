# Project Map

## Root
- `run.py` - application entrypoint
- `requirements.txt` - Python dependencies
- `.env.example` - environment variable template
- `README.md` - project overview and run instructions

## Backend
- `backend/app/main.py` - FastAPI app setup and router registration
- `backend/app/config.py` - settings and environment configuration
- `backend/app/database.py` - SQLAlchemy engine/session setup
- `backend/app/schemas.py` - API response/request schemas

### Services
- `backend/app/services/alert_agent.py` - alert agent, replay, stats, notifications
- `backend/app/services/llm_service.py` - LLM summary generation and assistant responses
- `backend/app/services/lpr_service.py` - license plate recognition pipeline
- `backend/app/services/police_gesture_service.py` - police gesture recognition
- `backend/app/services/owner_gesture_service.py` - owner gesture recognition

### Routers
- `backend/app/routers/monitor.py` - monitoring, logs, alerts, replay, SSE, assistant
- `backend/app/routers/lpr.py` - LPR endpoints
- `backend/app/routers/police_gesture.py` - police gesture endpoints
- `backend/app/routers/owner_gesture.py` - owner gesture endpoints
- `backend/app/routers/websocket.py` - websocket push endpoints
- `backend/app/routers/auth.py` - authentication endpoints

### Models
- `backend/app/models/logs.py` - system log table
- `backend/app/models/alerts.py` - alert event table
- `backend/app/models/records.py` - recognition records
- `backend/app/models/user.py` - user table

### Utilities
- `backend/app/utils/logger.py` - persistent logging helpers
- `backend/app/utils/auth.py` - authentication helpers
- `backend/app/utils/crypto.py` - encryption helpers
- `backend/app/utils/user_language.py` - user-facing alert phrasing

## Logs
- `backend/logs/app.log` - general runtime log
- `backend/logs/alerts.log` - warning/critical alerts
- `backend/logs/errors.log` - errors and stack traces

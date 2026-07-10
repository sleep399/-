from pathlib import Path

from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# OpenAI 兼容 LLM 厂商预设（.env 中 LLM_PROVIDER 选其一，再填 LLM_API_KEY 即可）
LLM_PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "openai": {
        "base": "https://api.openai.com/v1",
        "model": "gpt-3.5-turbo",
        "label": "OpenAI",
    },
    "qwen": {
        "base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-turbo",
        "label": "通义千问 (阿里云 DashScope)",
    },
    "deepseek": {
        "base": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "label": "DeepSeek",
    },
    "zhipu": {
        "base": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
        "label": "智谱 GLM",
    },
}


class Settings(BaseSettings):
    app_name: str = "车载视觉感知与人机交互系统"
    secret_key: str = "dev-secret-key-change-in-production"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    database_url: str = ""
    database_echo: bool = False
    odbc_driver: str = "ODBC Driver 17 for SQL Server"
    log_level: str = "INFO"

    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        db_path = self.data_dir / "app.db"
        return f"sqlite:///{db_path.as_posix()}"

    llm_provider: str = "qwen"                 # openai | qwen | deepseek | zhipu | custom
    llm_api_key: str = ""
    llm_api_base: str = ""                     # 留空则按 LLM_PROVIDER 自动选择
    llm_model: str = ""                        # 留空则按 LLM_PROVIDER 自动选择
    llm_timeout: float = 30.0                  # LLM API 超时（秒）
    llm_max_tokens: int = 800                  # 单次告警摘要最大 Token
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    alert_email_to: str = ""
    webhook_url: str = ""
    ccpd_data_path: str = "../CCPD-master"
    ctpgr_data_path: str = "../ctpgr-pytorch-master"
    hagrid_data_path: str = "../hagrid-master"
    aes_key: str = "0123456789abcdef0123456789abcdef"
    access_token_expire_minutes: int = 60 * 24
    gesture_hold_threshold: float = 0.8
    alert_failure_threshold: int = 5
    low_confidence_threshold: float = 0.4
    lpr_min_confidence: float = 0.5

    # ── 告警智能体增强配置 ──
    alert_window_seconds: int = 300            # 滑动窗口（秒），用于计算失败率
    alert_cooldown_seconds: int = 60           # 同类型告警冷却时间（秒），避免重复告警
    alert_config_cooldown_seconds: int = 3600  # 配置类告警冷却（启动/可选配置，避免刷屏）
    alert_token_warning_threshold: int = 80000 # Token 用量警告阈值
    alert_token_critical_threshold: int = 95000# Token 用量严重阈值
    alert_token_limit: int = 100000            # Token 配额上限
    alert_anomaly_rate_threshold: float = 0.3  # 异常比例阈值（如失败率 > 30%）
    alert_sse_enabled: bool = True             # 是否启用 SSE 推送
    alert_webhook_enabled: bool = False        # 是否启用 Webhook 推送（暂时关闭）
    alert_email_enabled: bool = False          # 是否启用邮件推送（暂时关闭）

    class Config:
        env_file = ".env"
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def base_dir(self) -> Path:
        return BASE_DIR

    @property
    def upload_dir(self) -> Path:
        p = self.base_dir / "uploads"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data_dir(self) -> Path:
        p = self.base_dir / "data"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def effective_llm_base(self) -> str:
        if self.llm_api_base:
            return self.llm_api_base.rstrip("/")
        preset = LLM_PROVIDER_PRESETS.get(self.llm_provider.lower(), {})
        return preset.get("base", LLM_PROVIDER_PRESETS["openai"]["base"]).rstrip("/")

    @property
    def effective_llm_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        preset = LLM_PROVIDER_PRESETS.get(self.llm_provider.lower(), {})
        return preset.get("model", LLM_PROVIDER_PRESETS["openai"]["model"])

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key.strip())

    @property
    def llm_provider_label(self) -> str:
        preset = LLM_PROVIDER_PRESETS.get(self.llm_provider.lower())
        if preset:
            return preset["label"]
        return "自定义 (custom)"


settings = Settings()

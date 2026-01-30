"""Application configuration using pydantic-settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Get the directory containing this config file (app/)
_APP_DIR = Path(__file__).parent
# Get the backend directory (parent of app/)
_BACKEND_DIR = _APP_DIR.parent
# .env file is in the backend directory
_ENV_FILE = _BACKEND_DIR / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/video_extract_pro",
        description="Async database URL",
    )
    database_sync_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/video_extract_pro",
        description="Sync database URL for Celery workers",
    )

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0")

    # LLM API Keys
    anthropic_api_key: str = Field(default="")
    openai_api_key: str = Field(default="")

    # LLM Configuration
    llm_provider: Literal["anthropic", "openai"] = Field(default="anthropic")
    claude_model: str = Field(default="claude-3-5-sonnet-20241022")
    openai_model: str = Field(default="gpt-4-turbo-preview")

    # Hugging Face
    hf_token: str = Field(default="", description="Hugging Face token for pyannote models")

    # Storage paths
    storage_path: Path = Field(default=Path("./storage"))
    video_storage_path: Path = Field(default=Path("./storage/videos"))
    clip_storage_path: Path = Field(default=Path("./storage/clips"))
    cache_path: Path = Field(default=Path("./storage/cache"))

    # ML Model settings
    whisper_model: str = Field(default="small")
    whisper_device: str = Field(default="cuda")
    whisper_compute_type: str = Field(default="float16")
    yolo_model: str = Field(default="yolov8x.pt")

    # Processing settings
    max_video_size_mb: int = Field(default=2000)
    max_video_duration_minutes: int = Field(default=180)
    analysis_timeout_seconds: int = Field(default=3600)

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    debug: bool = Field(default=False)
    cors_origins: list[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5500",
            "http://localhost:5500",
            "http://127.0.0.1:3000",
            "null",  # For file:// protocol
        ]
    )

    def ensure_storage_dirs(self) -> None:
        """Create storage directories if they don't exist."""
        for path in [
            self.storage_path,
            self.video_storage_path,
            self.clip_storage_path,
            self.cache_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.ensure_storage_dirs()
    return settings

"""Typed configuration; paths are anchored to storage_root, never the working directory."""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL, make_url

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")
    grobid_url: str = "http://localhost:8070"
    grobid_timeout: float = Field(default=180, gt=0)
    chunk_min_tokens: int = Field(default=200, ge=1)
    chunk_target_tokens: int = Field(default=300, ge=1)
    chunk_max_tokens: int = Field(default=400, ge=1)
    reranker_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    reranker_max_length: int = Field(default=512, ge=32, le=512)
    reranker_batch_size: int = Field(default=16, ge=1)
    reranker_revision: str | None = None
    generation_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    primera_revision: str | None = None
    qwen_revision: str | None = None
    primera_max_new_tokens: int = Field(default=256, ge=32, le=1024)
    primera_min_new_tokens: int = Field(default=56, ge=0, le=1024)
    primera_num_beams: int = Field(default=4, ge=1, le=16)
    primera_length_penalty: float = Field(default=2.0, gt=0, le=10)
    qwen_max_new_tokens: int = Field(default=384, ge=64, le=8192)
    generation_safety_margin: int = Field(default=256, ge=0, le=4096)
    generation_poll_seconds: float = Field(default=2.0, gt=0, le=60)
    generation_stale_minutes: int = Field(default=60, ge=5)
    zotero_rdf: Path = Path("data/raw_data/zotero_metadata.rdf")
    app_env: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    database_url: SecretStr | None = None
    postgres_host: str = "localhost"
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_db: str = "related_work"
    postgres_user: str = "related_work"
    postgres_password: SecretStr | None = None
    storage_root: Path = PROJECT_ROOT
    raw_pdfs_dir: Path = Path("data/raw_data/papers")
    processed_dir: Path = Path("data/processed")
    output_dir: Path = Path("data/outputs")
    model_cache_dir: Path = Path("models_cache/huggingface")
    qwen_lora_dir: Path = Path("models_cache/qwen_lora")

    @model_validator(mode="after")
    def validate_settings(self) -> "Settings":
        if self.database_url:
            try:
                url = make_url(self.database_url.get_secret_value())
                valid = url.drivername == "postgresql+psycopg" and bool(url.database)
            except Exception:
                valid = False
            if not valid:
                raise ValueError("DATABASE_URL must be a valid postgresql+psycopg URL")
        elif not self.postgres_password or not self.postgres_password.get_secret_value():
            raise ValueError("POSTGRES_PASSWORD or DATABASE_URL is required")
        if not self.storage_root.is_absolute():
            raise ValueError("STORAGE_ROOT must be absolute")
        if not self.chunk_min_tokens <= self.chunk_target_tokens <= self.chunk_max_tokens:
            raise ValueError("CHUNK_MIN_TOKENS must be <= CHUNK_TARGET_TOKENS <= CHUNK_MAX_TOKENS")
        for name in (
            "raw_pdfs_dir",
            "processed_dir",
            "output_dir",
            "model_cache_dir",
            "qwen_lora_dir",
            "zotero_rdf",
        ):
            path = getattr(self, name)
            setattr(self, name, (self.storage_root / path).resolve())
        return self

    @property
    def sqlalchemy_url(self) -> URL:
        if self.database_url:
            return make_url(self.database_url.get_secret_value())
        return URL.create(
            "postgresql+psycopg",
            username=self.postgres_user,
            password=self.postgres_password.get_secret_value(),
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )

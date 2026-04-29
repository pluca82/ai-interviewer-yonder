from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None
    """When False, omit response_format (e.g. Ollama OpenAI-compatible server)."""

    json_response_format: bool = Field(
        default=True,
        validation_alias="OPENAI_JSON_RESPONSE_FORMAT",
    )
    transcript_dir: Path = Path("storage/data")


@lru_cache
def get_settings() -> Settings:
    return Settings()

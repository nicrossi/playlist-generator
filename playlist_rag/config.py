from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=False, extra="ignore"
    )

    database_url: str = (
        "postgresql://playlist:password@localhost:5435/playlist_rag"
    )

    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3
    llm_max_retries: int = 3
    llm_timeout_seconds: int = 60

    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    indexer_version: str = "0.1.0"
    lyrics_excerpt_chars: int = 800
    batch_commit_size: int = 10


settings = Settings()
